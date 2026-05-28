from dataclasses import dataclass, field
import json, torch, random, tqdm, io, functools,os
from PIL import Image
from torch.utils.data import Dataset
from transformers import logging, AutoProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration
from torchvision.transforms.functional import pil_to_tensor
from transformers.feature_extraction_utils import BatchFeature
from collections import defaultdict
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List
from qwen_vl_utils.vision_process import smart_nframes, process_vision_info, FPS, VIDEO_TOTAL_PIXELS, VIDEO_MIN_PIXELS, FPS_MAX_FRAMES, FORCE_QWENVL_VIDEO_READER
from streaming_vlm.utils.get_qwen_range import get_qwen_range
from transformers import set_seed

class mute_stderr_ffmpeg:
    def __enter__(self):
        self._stderr_fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self
    def __exit__(self, *exc):
        os.dup2(self._stderr_fd, 2)
        os.close(self._devnull)
        os.close(self._stderr_fd)

logger = logging.get_logger(__name__)

def _video_tensor_to_np_frames(video):
    """Convert a (T,3,H,W) torch.Tensor of frames to list[np.ndarray (H,W,3) uint8].

    Frames are expected to come from livecc_utils._spatial_resize_video, which
    yields float32 (T,3,H,W) tensors in the [0, 255] range (BICUBIC resize of
    uint8 frames, not normalized). uint8 inputs are also accepted unchanged.
    """
    if isinstance(video, torch.Tensor):
        assert video.ndim == 4 and video.shape[1] == 3, (
            f"_video_tensor_to_np_frames expects (T,3,H,W) tensor, got shape {tuple(video.shape)}"
        )
        if video.dtype != torch.uint8:
            # float32 [0, 255] from BICUBIC resize: clamp to handle overshoot,
            # round to avoid biased truncation, then cast.
            video = video.clamp(0, 255).round().to(torch.uint8)
        # (T,3,H,W) -> (T,H,W,3)
        arr = video.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [arr[i] for i in range(arr.shape[0])]
    return video  # already list[np.ndarray] or list[PIL.Image] etc.

@dataclass
class DataArguments:
    train_annotation_paths: list[str] = None
    initial_fps_frames: int = int(FPS)
    streaming_fps_frames: int = int(FPS)
    with_context: bool = False
    text_sink:int = 0
    text_sliding_window:int = 0

@dataclass
class EvalDataArguments:
    eval_annotation_paths: list[str] = None

def readlastline(path: str):
    """Efficiently read the last line of a file."""
    with open(path, "rb") as f:
        f.seek(-2, 2)
        while f.read(1) != b"\n":
            f.seek(-2, 1)
        return f.readline()

def bytes_to_pil(image_bytes):
    """Convert bytes to a PIL Image object."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'P':
        image = image.convert('RGBA')
    return image.convert('RGB')

def get_phrase_before_timestamp(text_stream, timestamp, start_from: int = 0):
    """From a word stream with timestamps, extract the phrase consisting of all words whose end time is before the given timestamp."""
    phrase = ''
    i = 0
    for i, (ws, we, word) in enumerate(text_stream[start_from:]):
        if timestamp >= we:
            phrase += ' ' + word.strip()
            if i == len(text_stream[start_from:]) - 1:
                i += 1
                break
        else:
            break
    return phrase, i + start_from

class LMMDataset(Dataset):
    """PyTorch dataset for multimodal large language models."""
    def __init__(
        self, *, train_annotation_paths: list[str] = None, eval_annotation_paths: list[str] = None, processor: AutoProcessor, tokenizer = None,
        initial_fps_frames: int = DataArguments.initial_fps_frames, streaming_fps_frames: int = DataArguments.streaming_fps_frames, 
        with_context: str = DataArguments.with_context, return_conversation: bool = False,text_sink:int = DataArguments.text_sink,
        text_sliding_window:int = DataArguments.text_sliding_window,
        **kwargs
    ):
        super().__init__()
        self.return_conversation = return_conversation
        self.handles = []
        if eval_annotation_paths is not None:
            self.is_eval_dataset = True
            annotation_paths = eval_annotation_paths
        else:
            self.is_eval_dataset = False
            annotation_paths = train_annotation_paths
            
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), "Please organize annotation data as JSONL (one sample per line) and store the final line as the seek index."
            root, fname = os.path.split(annotation_path)
            stem = fname.replace("_with_seeks", "").rsplit(".jsonl", 1)[0]
            seek_path = os.path.join(root, f"{stem}_seeks.jsonl")

            logger.warning(f"Loading {annotation_path}")
            logger.warning(f"Loading seek index from {seek_path}")

            with open(seek_path) as f:
                seeks = json.load(f)     
                
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f"Successfully loaded {annotation_path}")

        proc_name = processor.__class__.__name__
        if 'Qwen2VL' in proc_name or 'Qwen2_5_VL' in proc_name:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'Qwen2'
        elif 'LlavaOnevision2' in proc_name:
            # Same Qwen tokenizer family; im_start/im_end/assistant IDs are identical.
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'LlavaOnevision2'
        else:
            raise NotImplementedError(f"Video preprocessing for {proc_name} is not implemented")

        self.processor = processor
        self.with_context = with_context
        self.initial_fps_frames = initial_fps_frames
        self.streaming_fps_frames = streaming_fps_frames
        self.text_sink = text_sink
        self.text_sliding_window = text_sliding_window
    
    def load_conversation(self, index):
        """Load a single conversation by index."""
        annotation_path, seek = self.handles[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line)
        return line

    def preprocess_image(self, element: dict):
        """Preprocess image data."""
        if hasattr(self, 'remote_loader'):
            return Image.open(self.remote_loader(element['image']))
        return element['image']
    
    def preprocess_video(self, element: dict):
        """Preprocess video data."""
        if 'pos' in element:
            positions = [0] + element['pos']
            nframes = smart_nframes(element, total_frames=len(positions) - 1, video_fps=FPS)
            sampler = torch.linspace(0, len(positions) - 2, nframes).round().long()
            data_bytes = self.remote_loader(element['video'], length_check=True, return_io=False)
            video = torch.stack([pil_to_tensor(bytes_to_pil(data_bytes[positions[i]:positions[i+1]])) for i in sampler])
            video = _spatial_resize_video(video)
            return video
        return element['video']

    def preprocess_text(self, element: str):
        """Preprocess text data."""
        return element['text']

    def preprocess_conversation_stream(self, conversation: list):
        """Simulate converting timestamped conversation into a streaming multi-turn dialogue."""
        user_message, assistant_message = conversation
        user_content, assistant_content = user_message['content'], assistant_message['content']

        user_video_dict, user_query_dict = user_content
        video_start = user_video_dict['video_start']
        video_end = user_video_dict['video_end']
        
        assert 'video' in user_video_dict, 'Please check your data: the first user content must contain video information'

        assistant_text_stream = assistant_message['content'][0]['text_stream']
        qa_stream = assistant_message['content'][0]['qa_stream'] if 'qa_stream' in assistant_message['content'][0] else []
        
        with mute_stderr_ffmpeg():
            clip, _, clip_pts = _read_video_decord_plus(
                user_video_dict, return_pts=True, strict_fps=True
            )
        clip = _spatial_resize_video(clip)

        start_timestamp, end_timestamp = video_start, video_start + self.initial_fps_frames / FPS

        phrase, next_start_from = get_phrase_before_timestamp(
            assistant_text_stream,
            end_timestamp
        )

        if len(qa_stream) > 0 and start_timestamp < qa_stream[0][1] and end_timestamp >= qa_stream[0][1]:
            question = qa_stream[0][2]
            answer = qa_stream[0][3]
            qa_stream = qa_stream[1:]
        else:
            question = ''
            answer = ''
        
        user_content = [
                        {'type': 'text',  'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s' + f'{question}'},
                        {'type': 'video', 'video': clip[:self.initial_fps_frames]},
                    ]
        assistant_content = [{'type': 'text', 'text': answer + '\n' + phrase + ' ...'}]
        conversation = [
            {
                'role': 'user',
                'content': user_content
            },
            {
                'role': 'assistant',
                'content': assistant_content
            }
        ] 
        frames_list = [clip[:self.initial_fps_frames]]
        
        for i in range(self.initial_fps_frames, len(clip), self.streaming_fps_frames):
            start_timestamp, end_timestamp = video_start + i / FPS, video_start + (i + self.streaming_fps_frames) / FPS
            
            phrase, next_start_from = get_phrase_before_timestamp(
                assistant_text_stream,
                end_timestamp,
                start_from=next_start_from
            )
            if len(qa_stream) > 0 and start_timestamp < qa_stream[0][1] and end_timestamp >= qa_stream[0][1]:
                question = qa_stream[0][2]
                answer = qa_stream[0][3]
                qa_stream = qa_stream[1:]
            else:
                question = ''
                answer = ''
            
            frames = clip[i : i + self.streaming_fps_frames]

            user_content = [
                    {'type': 'text',  'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s {question}'},
                    {'type': 'video', 'video': frames},
                ]
            assistant_content = [{'type': 'text', 'text': answer + '\n' + phrase + ' ...'}]

            conversation.extend([
                {
                    'role': 'user',
                    'content': user_content
                },
                {
                    'role': 'assistant',
                    'content': assistant_content
                }
            ])
            frames_list.append(frames)

        return conversation, frames_list

    def getitem(self, index, return_text=False):
        """Core logic to get and preprocess a single data item."""
        conversation = self.load_conversation(index)

        special_process_for_stream, image_inputs, video_inputs = False, None, None
        previous_text = ''
        for message in conversation:
            if message['role'] == 'user':
                for element in message['content']:
                    if 'previous' in element:
                        previous_text = element['previous']
                        element['previous'] = ''
                    if hasattr(self, 'remote_loader'):
                        element['remote_loader'] = self.remote_loader
                    modal = element['type']
                    element[modal] = getattr(self, f'preprocess_{modal}')(element)
                    if isinstance(element[modal], torch.Tensor):
                        if video_inputs is None:
                            video_inputs = [element[modal]]
                        else:
                            video_inputs.append(element[modal])
            else:
                for element in message['content']:
                    special_process_for_stream = 'text_stream' in element
                    break
        
        if not os.path.exists(conversation[0]['content'][0]['video']):
            if os.path.exists(os.path.join(os.environ['DATASET_PATH'], conversation[0]['content'][0]['video'])):    
                conversation[0]['content'][0]['video'] = os.path.join(os.environ['DATASET_PATH'], conversation[0]['content'][0]['video'])
            else:
                raise ValueError(f"Video {conversation[0]['content'][0]['video']} not found")
        if special_process_for_stream:
            conversation, video_inputs = self.preprocess_conversation_stream(conversation)
            image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)

        conversation = [{"role": "previous text", "content": previous_text}] + conversation

        if return_text:
            return conversation
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, return_tensors='pt')
        
        if self.model_base == 'LlavaOnevision2' and video_inputs is not None:
            # LlavaOnevision2VideoProcessor accepts file path / list[PIL] / list[np.ndarray] only.
            video_inputs = [_video_tensor_to_np_frames(v) for v in video_inputs]
            # The processor's video path expects one <|video_pad|> per video and one frame list per video.
            # Use num_frames=None so the processor takes len(frames) directly (no resampling).
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                num_frames=None,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )

        if self.text_sink != 0 or self.text_sliding_window != 0:
            previous_text_start_idx, previous_text_end_idx = self.get_range(inputs.input_ids, 'previous text', 0, contain_lf=True)
            need_truncate = previous_text_start_idx + self.text_sink + self.text_sliding_window <= previous_text_end_idx + 1
            if need_truncate:
                truncate_start = previous_text_start_idx + self.text_sink
                truncate_end = previous_text_end_idx - self.text_sliding_window
                inputs['input_ids'] = torch.cat([inputs.input_ids[:,:truncate_start],inputs.input_ids[:,truncate_end+1:]],dim=1).contiguous()
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = torch.cat([inputs.attention_mask[:,:truncate_start],inputs.attention_mask[:,truncate_end+1:]],dim=1).contiguous()

        input_ids = inputs.input_ids
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx+3:im_end_idx+1] = input_ids[sample_idx, im_start_idx+3:im_end_idx+1]

        inputs['labels'] = labels
        if self.return_conversation:
            inputs['conversation'] = conversation
            inputs['start_timestamp'] = conversation[0]['content'][0]['video_start']

        return inputs

    def __getitem__(self, index):
        """Dataset standard method with retry."""
        try: 
            return self.getitem(index) 
        except Exception as e:
            logger.warning(
                f"{'Training' if not self.is_eval_dataset else 'Eval'}: bug at video: "
                f"{self.load_conversation(index)[0]['content'][0]['video']}"
                f"{e}"
            )
        return self.__getitem__( index*13 %len(self.handles))

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1, "batch size must be 1"
        return batched_inputs[0]

    def __len__(self):
        """Return the total number of samples."""
        return len(self.handles)

def get_ground_truth(dataset, idx, processor):
    video_text = dataset.getitem(idx, return_text=True)
    ground_truths = []
    for round in video_text:
        if round['role'] == 'assistant':
            ground_truth = round['content'][0]['text']
            ground_truths.append({'ground_truth':ground_truth})
        else:
            continue
    return ground_truths

if __name__ == "__main__":
    import argparse
    set_seed(1314)
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=List[str], default=[]) 
    args.add_argument('--model_base', type=str, default='Qwen',choices=['Qwen'])
    args.add_argument('--idx', type=int, default=None)
    args.add_argument('--text_sink', type=int, default=512)
    args.add_argument('--text_sliding_window', type=int, default=512)
    args = args.parse_args()

    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side='right') 
    if args.idx is not None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',trust_remote_code=True,device_map='auto',attn_implementation='flash_attention_2')
        
    dataset = LMMDataset(
        train_annotation_paths=args.data_path, 
        tokenizer=None,
        processor=processor,
        text_sink=args.text_sink,
        text_sliding_window=args.text_sliding_window,
        with_context=False,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=128, collate_fn=dataset.data_collator)
    
    if args.idx is not None:
        data = dataset[args.idx]
        breakpoint()

        model.generate(input_ids=data.input_ids.to('cuda'),media=data.media,media_config=defaultdict(dict), generation_config=model.default_generation_config)
        breakpoint()
    else:
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            pass
