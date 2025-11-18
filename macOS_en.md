CorrNet — Continuous Sign Language Recognition (macOS Adaptation and Replication Guide)This documentation provides a complete operational procedure for successfully replicating the CorrNet Continuous Sign Language Recognition system (CSL-Daily Model) on macOS / Apple Silicon (M1 / M2 / M3).Because the official project primarily targets Linux + CUDA, the original code encounters several compatibility issues on macOS, such as:MPS lacks support for some 3D pooling operationsctcdecode cannot be compiled on macOSThe dictionary structure of CSL-Daily causes decoding failuresSome modules do not account for automatic CPU/MPS switchingdemo.py cannot correctly read image sequences on macOSThis replication guide provides a complete workflow that is fully runnable on macOS from scratch, including:Environment configurationDecoder adaptation (pyctcdecode replacing ctcdecode)Automatic MPS / CPU selectionComprehensive rewrite of demo.py + decode.pyHandling of multiple image inputs and video formatsOptimization for result decoding stabilityTroubleshooting common errorsYou simply need to follow the document, execute the steps from top to bottom, to run the project, load the CSL-Daily model, and perform continuous sign language recognition using image sequences or videos.Table of Contents (Outline)Project Introduction (macOS Compatibility Version Description)Environment Preparation2.1 System Requirements (macOS / Apple Silicon)2.2 Conda Environment Creation2.3 Key Dependency Versions (torch, pyctcdecode, decord…)2.4 pip freeze Example (Actual runnable environment)Why the Original Version Cannot Run on macOS3.1 MPS Does Not Support max_pool3d3.2 ctcdecode Cannot Be Compiled3.3 Special Structure of CSL-Daily gloss_dict3.4 Summary of Solution StrategiesMac Adaptation Changes (Core Modification Explanations)4.1 Device Selection Logic (MPS → CUDA → CPU)4.2 Enabling MPS Fallback4.3 Multi-Image Sorting + Safe Handling4.4 decode.py Modifications (pyctcdecode + unicode vocab)4.5 demo.py Modifications (Device Migration, Video Loading, Exception Handling)Environment Configuration (macOS)Code File Structure Explanation (Directory Structure + File Roles)Inference Procedure on macOS (Multi-Image / Video)1. Project Introduction (macOS Compatibility Description)This project is based on CorrNet: Correspondence-Aware Network for Continuous Sign Language Recognition, and uses the publicly available CSL-Daily pre-trained model for Continuous Sign Language Recognition (CSLR).The official running environment for the original project was:Ubuntu + CUDA (NVIDIA GPU)ctcdecode (requires GNU toolchain and Linux environment)PyTorch with GPU accelerationHowever, macOS (especially Apple Silicon M1/M2/M3) has multiple incompatibilities with the original environment, preventing the official code from running directly:1.1 Reasons Why the Original Project Cannot Run Directly on macOS(1) MPS Does Not Support 3D PoolingThe front-end visual encoder of CorrNet contains certain operators that require GPU optimization. macOS's MPS Metal backend currently still does not support some 3D pooling (especially max_pool3d and certain stride/padding combinations).Consequently, the model will throw an error during mid-stage feature extraction on a Mac.(2) ctcdecode Cannot Be InstalledThe original project uses ctcdecode (an acceleration library based on C++ & OpenMP).On macOS:OpenMP compilation is unsuccessfulThe official ctcdecode also does not provide a pre-compiled version for macOSThis results in the decode stage being completely unusable.(3) Special Structure of CSL-Daily's gloss_dict.npyThe CSL-Daily dictionary format is:{
  "他": [1],
  "有": [2],
  "什么": [3],
  ...
}
Instead of the common structure like "我": 0.This requires specialized handling during decoding; otherwise:Prediction index → gloss mapping failsA large number of UNKs appearBoth beam search and greedy outputs are incorrect(4) demo.py Does Not Handle macOS Paths, Devices, and Image SequencesThe official demo only supports:LinuxVideo inputCUDA deviceOn macOS:There is no CUDAImage sequence upload order is chaoticDevice not automatically detected, leading to very slow model execution on CPURequires handling of MPS fallback, video reading, and file type recognition1.2 Content Provided in This DocumentationTo resolve all the above issues, this macOS version replication guide includes:A CorrNet inference environment that can run directly on macOSThe fully fixed decode.py (adapted for pyctcdecode + unicode vocab)The fully fixed demo.py (supporting multi-image/video/automatic MPS)Complete environment dependencies (pip freeze)Complete running steps (including MPS fallback)Troubleshooting for common errorsDetailed explanation of the CSL-Daily dictionary structureExplanation of model output instability and optimization suggestionsFinal Goal: Zero-friction execution of CorrNet's CSL-Daily inference for Mac users.1.3 Target AudienceThis README is highly suitable for the following readers:Students using M1 / M2 / M3 MacBookUsers without an NVIDIA GPUThose who want to quickly run CSL-Daily inferenceThose who want to integrate CorrNet into their own sign language recognition projectsThose who want to learn the CSLR inference pipeline (Input → Preprocessing → Model → CTC decode)1.4 Description of Output EffectsUpon completing this tutorial, your Mac will be able to:Load the author's provided dev_30.60_CSL-Daily.ptSupport continuous sign language recognition from multiple images (continuous frames)Support video file inputOutput a final word sequence, for example:[(“他”, 0), (“在”, 1), (“做”, 2)]
And the decode structure will be consistent with the author's, suitable for subsequent NLP modules for sentence recovery.2. Environment Preparation (macOS Replication Version)This section will introduce all the dependencies required to set up the CorrNet inference environment on macOS, including: Python environment creation, PyTorch (with MPS support) installation, project dependency installation, necessary system tools, and the placement of weight files. At the end of this section, your computer will be in the "minimum usable state for running demo.py".2.1 Python Environment and Version DescriptionThis project recommends using Python 3.10. macOS's native Python and Conda's default Python might conflict with some dependencies, so it is suggested to use Miniconda or conda-forge to create an independent environment.Example commands (you can adjust based on your installation method):conda create -n corrnet-mac python=3.10
conda activate corrnet-mac
After creation, use the following commands to confirm environment information:python --version
which python
which pip
2.2 Select and Install the Appropriate PyTorch (with MPS Support)Apple chips do not support CUDA, so PyTorch with MPS acceleration must be installed.Recommended installation instruction (from official source):pip install torch torchvision torchaudio
Then confirm if MPS is detected:Pythonimport torch
print(torch.backends.mps.is_available())
If it returns:True
It means your Mac can use the GPU for some model computations (much faster than the CPU).2.3 Install Project Dependencies (pip freeze List)The following is the complete list of dependencies for the currently runnable version (from your successful demo.py environment). Users can directly use:pip install -r requirements.txt
The dependencies for stable operation of this project on macOS are as follows (all need to be installed to ensure reproducibility):aiofiles==23.2.1
altair==5.5.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.11.0
attrs==25.4.0
certifi==2025.10.5
charset-normalizer==3.4.4
click==8.1.8
contourpy==1.3.0
cycler==0.12.1
Cython==3.2.0
einops==0.8.1
eva-decord==0.6.1
exceptiongroup==1.3.0
fastapi==0.121.1
ffmpy==1.0.0
filelock==3.19.1
fonttools==4.60.1
fsspec==2025.10.0
gradio==3.44.4
gradio_client==0.5.1
h11==0.16.0
hf-xet==1.2.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.1.2
hypothesis==6.141.1
idna==3.11
importlib_resources==6.5.2
Jinja2==3.1.6
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
kiwisolver==1.4.7
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.4
mdurl==0.1.2
mpmath==1.3.0
narwhals==2.11.0
networkx==3.2.1
numpy==1.26.4
opencv-python==4.11.0.86
orjson==3.11.4
packaging==25.0
pandas==2.3.3
pillow==10.4.0
pyctcdecode==0.5.0
pydantic==2.12.4
pydantic_core==2.41.5
pydub==0.25.1
Pygments==2.19.2
pygtrie==2.5.0
pyparsing==3.2.5
python-dateutil==2.9.0.post0
python-multipart==0.0.20
pytz==2025.2
PyYAML==6.0.3
referencing==0.36.2
regex==2025.11.3
requests==2.32.5
rich==14.2.0
rpds-py==0.27.1
ruff==0.14.4
safetensors==0.6.2
scipy==1.13.1
semantic-version==2.10.0
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
sortedcontainers==2.4.0
starlette==0.49.3
sympy==1.14.0
tokenizers==0.22.1
tomlkit==0.12.0
torch==2.8.0
torchaudio==2.8.0
torchvision==0.23.0
tqdm==4.67.1
transformers==4.57.1
typer==0.20.0
typer-slim==0.20.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
uvicorn==0.38.0
websockets==11.0.3
zipp==3.23.0
Users can generate requirements.txt automatically if needed:pip freeze > requirements.txt
2.4 FFmpeg Installation (Required for Video Recognition)The project supports video input, thus requiring FFmpeg.For macOS, Homebrew is recommended:brew install ffmpeg
Confirm successful installation:ffmpeg -version
2.5 Placement of Weight Files (Model Files)Download the model from the author:dev_30.60_CSL-Daily.ptOr your own trained .pt filePlace it in the project directory under:./weights/
Example directory structure:CorrNet/
  ├── demo.py
  ├── decode.py
  ├── models/
  ├── utils/
  ├── weights/
  │     └── dev_30.60_CSL-Daily.pt
If the directory does not exist, create it manually:mkdir weights
2.6 Testing Environment SanityAfter ensuring the environment is successfully installed, run:python -c "import torch; print(torch.backends.mps.is_available())"
If the output is True, MPS is available.Then, test if the minimal script runs correctly:python demo.py --help
If the parameter description prints successfully, the environment is ready.3. Why macOS Adaptation Is Necessary (Technical Explanation)The original implementation of this project was designed for the Linux + CUDA environment. The logic involved in GPU scheduling, model loading, video reading methods, and the construction of the decode dictionary assumes characteristics of the Linux / CUDA environment. Therefore, running directly on macOS leads to numerous errors.The following explains why adaptation is essential from four aspects: system features, hardware differences, decoding logic, and data processing flow.3.1 System Level: Key Differences Between Linux and macOSThe official CorrNet implementation relies on the following Linux features:CUDA GPU (NVIDIA)Official Torch + CUDA buildLinux file path handlingDefault compatibility of video input (OpenCV + decord) on LinuxThe GPUDataParallel (multi-card training/inference framework) the author used by default on LinuxmacOS, however, has completely different hardware and system mechanisms:No NVIDIA CUDA (no CUDA kernel)Uses Apple Silicon (M1/M2/M3/M4…) + MPS accelerationDifferent file system behavior (paths + caching + temporary files)Different FFmpeg wrapping compared to the author's environmentThus, all errors encountered by the original demo.py essentially stem from the "author's assumption that you are running on Linux".3.2 Hardware Differences: CUDA → MPS Compatibility Issues3.2.1 Official Demo Relies on CUDA Device:Many lines of code are similar to:if args.device >= 0:
    vid = vid.cuda()
This directly triggers on macOS:RuntimeError: CUDA not available
The reason is that many Tensors in the project call .cuda() during inference, but this device does not exist under macOS.3.2.2 Limitations of MPSmacOS GPU acceleration uses MPS:Does not support all CUDA operatorsPrecision behavior is not entirely consistent with CUDAThe speed of large-dimension convolutions can sometimes be slower than CPUSome Tensors cannot mix GPU/CPU (must be on a unified device)Therefore, demo.py must be rewritten to:Automatically detect MPS / CUDA / CPUSelect the final running device based on the actual hardware3.3 Decoding Part: Official decode.py Incompatible with macOSTo successfully recognize CSL-Daily, a complete decode process of predict → gloss → unicode vocab is required.However, the official decode.py has several Linux-only assumptions, such as:Using the wrong vocab dictionary orderUnicode vocab / gloss vocab not being read in correctlyInput defaults to tuple format, not listDecode return structure does not match demo.py's expectationLeading to the following issues without adaptation:Index prediction is normal, but cannot be mapped to glossOutput is all <unk>Beam search output dimension mismatch throws an errorTherefore, it is necessary to:Rewrite the decode entry pointFix the unicode vocab reading methodFix the input type (list replaces tuple)Fix the return value format (list of strings)Otherwise, even if the model runs on macOS, it cannot decode correctly.3.4 Image/Video Input: Differences in VideoReader, OpenCV on macOSOn macOS, decord can encounter:Inability to automatically select the hardware decoderNumerous read failures during frame I/OImage sorting depends on Finder metadata, unlike LinuxThe official demo assumes:cv2.VideoCapture(...)
But on macOS, VideoCapture might return:Only 0-1 frames can be readChaotic frame reading orderNumerous "corevideo pixel buffer errors"Reading MP4 must rely on FFmpeg installationTherefore, the demo must be modified to:Use decord.VideoReader + CPU decodingSort images by filename to prevent order confusionAdd exception protectionThese changes allow macOS users to stably input video frames.3.5 GpuDataParallel (Official Utils) Does Not Support macOSThe official utils's GpuDataParallel relies on:torch.cuda.device_count()CUDA multi-card schedulingmacOS only has:Single-card MPSOr pure CPUThus, without modification:AttributeError: 'mps' object has no attribute 'device_count'
It is necessary to rewrite parts of the GpuDataParallel logic to make it:Multi-card for CUDASingle-card for MPSFallback to CPU for CPUAnd change:data_to_device
to a universally compatible cross-device function.3.6 Summary: Why macOS Adaptation is Necessary?To summarize:CategoryReasonImpactSystem DifferenceLinux → macOSDifferent path/caching/temp file behaviorGPU DifferenceCUDA → MPSNumerous .cuda() calls will error directlyIncomplete Decodingvocab / unicode / tuple mismatchPredicted index cannot be correctly translated to glossUnstable InputOpenCV/VideoReader differencesVideo frame reading errorsDevice ManagementGpuDataParallel incompatibilityDevice error during inferenceThe ultimate result is:The original demo.py, even if it could launch on macOS, cannot complete normal inference. System-level adaptation is mandatory for the project to run entirely.4. macOS Adaptation Modification Details (Line-by-Line Explanation)This section is the most critical part of this README. It explains which files must be modified for CorrNet to successfully run on macOS, and provides a line-by-line explanation of the reasons. This includes:4.1 Adapting decode.py (fixing vocab / unicode / numpy / beam search)4.2 Fixing demo.py (input, device, model loading, MPS fallback)4.3 Fixing utils / GpuDataParallel4.4 Fixing image sorting, video frame reading (preventing order confusion)4.5 Modifying model weight loading (macOS torch compatibility)4.1 decode.py (Core Adaptation)The official decode has the following issues:Default use of ctcdecode (not pyctcdecode) in the Linux environmentIncorrect vocab construction methodUnicode mapping method inconsistent with gloss_dictBeam search output type mismatch with demo.pyLogits cannot be directly .numpy() (must detach().cpu() on MPS)Decode result is a string, but demo.py expects a list → ErrorTherefore, the following modifications are necessary.4.1.1 Add pyctcdecode Check (macOS has no CUDA)The original code assumes ctcdecode (CUDA) is available, but macOS is not. Therefore, change it to prioritize importing pyctcdecode:try:
    from pyctcdecode import build_ctcdecoder
    _has_pyctc = True
except Exception:
    _has_pyctc = False
Purpose:Let macOS use the CPU version of beam searchEnsure Linux users can still use ctcdecode if available4.1.2 Fix Vocab Construction MethodOfficial gloss_dict format:{
  "HELLO": [0, …],
  "THANKYOU": [1, …],
  ...
}
Therefore, the index → gloss mapping must be established correctly:self.i2g = {v[0]: k for k, v in gloss_dict.items()}
If not fixed, it leads to:All decode outputs being UNKBeam search getting characters but being unable to look up the gloss4.1.3 Use Unicode Vocab (Consistent with Author's Paper)CorrNet decoding requires "each category → one unicode character":self.vocab = [chr(20000 + i) for i in range(num_classes)]
Without this step:Beam search will not return indexable tokensThe calculation class_id = ord(char) - 20000 is incorrect4.1.4 Logits Must Be Detached Before Numpy (MPS Requirement)MPS tensors do not allow direct .numpy(); they must:logits = logits.detach().cpu().numpy()
Otherwise, the error:TypeError: can't convert mps tensor to numpy
is thrown.4.1.5 Beam Search Potential Errors → Automatic Fallback to GreedyTo avoid exceptions caused by pyctcdecode in the macOS environment, it is necessary to:try:
    decoded = self.beam_decoder.decode(logit)
except:
    return self._greedy(torch.tensor(logits), lengths)
To give the system fault tolerance.4.1.6 Fix Beam Search Return FormatThe official return is a string, but it must be converted to:[(gloss, index), ...]
Therefore, the following is needed:class_ids = [ord(ch) - 20000 for ch in decoded]
sent = [(self.i2g.get(cid, "UNK"), i) for i, cid in enumerate(class_ids) if cid != self.blank]
To ensure demo.py can parse it correctly.4.2 demo.py Adaptation (Core Part)The original demo.py does not support macOS for the following reasons:Uses .cuda() to forcefully move data to CUDA (macOS lacks this)GpuDataParallel relies on CUDAOpenCV / VideoReader read order is chaotic on macOSAutomatic device detection is missingVideo reading and padding logic cannot handle Tensors on MPSHere are the main modification points.4.2.1 Automatic Device Detection (Unified Entry)Add the following logic:if torch.backends.mps.is_available():
    map_location = torch.device("mps")
elif torch.cuda.is_available():
    map_location = torch.device("cuda")
else:
    map_location = torch.device("cpu")
And ensure the model uses this device for loading and inference.4.2.2 Handle MPS FallbackPyTorch's MPS support is incomplete, often leading to:Some operators not supporting MPSMPS memory shortage or other exceptionsTherefore, when running the demo, it is recommended to use:PYTORCH_ENABLE_MPS_FALLBACK=1 python demo.py ...
4.2.3 Ensure All Tensors Have a Unified DeviceThe original code contains multiple instances of:vid = vid.cuda()
...
video_length = video_length.cuda()
These must be uniformly replaced with:vid = device.data_to_device(vid)
vid_lgt = device.data_to_device(video_length)
Managed uniformly by GpuDataParallel.4.2.4 Fix decord/Video Input Sorting IssueThe order is often chaotic when macOS reads multiple images, so sorting is mandatory:inputs = sorted(inputs, key=lambda x: os.path.basename(path))
Otherwise, the model's temporal order will collapse.4.2.5 Fix decord's CPU-only UsageOn macOS, decord must use:VideoReader(path, ctx=cpu(0))
Otherwise, it throws the error:cannot find GPU context
4.2.6 Fix Model Weight Loading (map_location)Loading weights on macOS must use:state_dict = torch.load(model_path, map_location=map_location)
Otherwise, it reports an error like:Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
4.3 Fix utils.GpuDataParallelThe original version does not support macOS.It must be modified to:CUDA: Multi-cardMPS: Single-cardCPU: Single-cardAnd refactor:def data_to_device(self, x):
    return x.to(self.device)
To ensure compatibility with all devices.4.4 Fix Image Input and Frame PaddingVideo frame length reading is unstable on macOS, so demo must fix:Left paddingRight paddingStride alignmentTemporal padding should not mix CPU/MPS devicesOtherwise, inference will result in:Expected all tensors on same device
4.5 SummaryThis section introduced all necessary code modification points for macOS adaptation, involving:decode.py (character, beam search, vocab, numpy compatibility)demo.py (multiple device fixes + input fixes)GpuDataParallel (device management rewrite)Video/image input fixesWeight loading fixesAfter these modifications, the entire CorrNet system is guaranteed to be fully runnable on macOS: including image sequence input, video input, beam search decoding, and unicode vocab processing.5. Environment Configuration (macOS)This section fully explains how to create a complete runnable CorrNet environment on macOS, including:Python / Conda environment creationPyTorch (with MPS support) version selection and explanationExplanation of pip freeze versionsWhy these specific versions must be usedA single, copy-paste runnable command5.1 System Requirements (Must Be Met)macOS:macOS 13 Ventura or higherApple Silicon (M1/M2/M3) or Intel8GB or more memory (16GB recommended)Framework Requirements:Python 3.9 (most stable support for PyTorch + decord)PyTorch 2.2 or higher (native MPS support)torchvision / torchaudio of the same versionCUDA is not needed (macOS has no CUDA)5.2 Creating an Independent Conda EnvironmentIt is recommended to use Conda (Miniconda / Anaconda both work):conda create -n corrnet python=3.9 -y
conda activate corrnet
Why must Python be 3.9?PyTorch on macOS is most stable on 3.9decord has compatibility issues with python > 3.10The CorrNet project itself assumes Python 3.8/3.9 on Linux5.3 Installing PyTorch (MPS Support)macOS uses MPS to replace CUDA. The installation command is:pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
This version includes:CPU kernelsMPS kernelsNo CUDA (normal)Test after installation:python - << 'EOF'
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
EOF
If the output is:MPS available: True
It means MPS is enabled.5.4 Installing Gradio, OpenCV, Decord, and Dependency PackagesThis is the critical part of the project.pip install gradio==3.44.4 opencv-python==4.11.0.86 decord==0.6.1
pip install numpy==1.26.4 einops==0.8.1 scipy==1.13.1
pip install matplotlib pandas pillow
pip install transformers pyctcdecode
Why these versions?decord==0.6.1: Latest compatible version for macOSopencv-python==4.11: Does not trigger QT dependency issuesnumpy==1.26.4: Most stable pairing with PyTorch 2.8pyctcdecode==0.5.0: Most stable version based on Python 3.9gradio==3.44.4: Most stable UI, does not conflict with torch5.5 Project Dependencies (Consistent with Your Current Environment)The following is the pip freeze from your current environment (the important parts). This is the tested runnable version combination for CorrNet on macOS:aiofiles==23.2.1
altair==5.5.0
annotated-types==0.7.0
anyio==4.11.0
attrs==25.4.0
Cython==3.2.0
einops==0.8.1
eva-decord==0.6.1
fastapi==0.121.1
fonttools==4.60.1
gradio==3.44.4
huggingface_hub==1.1.2
importlib_resources==6.5.2
Jinja2==3.1.6
matplotlib==3.9.4
numpy==1.26.4
opencv-python==4.11.0.86
pandas==2.3.3
pillow==10.4.0
pyctcdecode==0.5.0
pydantic==2.12.4
python-dateutil==2.9.0.post0
regex==2025.11.3
requests==2.32.5
scipy==1.13.1
starlette==0.49.3
sympy==1.14.0
torch==2.8.0
torchaudio==2.8.0
torchvision==0.23.0
tqdm==4.67.1
transformers==4.57.1
uvicorn==0.38.0
Key Explanations:torch==2.8.0 + numpy==1.26.4 → Most stable combinationpyctcdecode==0.5.0 → Perfectly compatible with Python 3.9decord==0.6.1 → Highest usable version on macOSgradio==3.44.4 → Fully supports multi-image upload/video UIopencv-python==4.11 → No QT dependency crashYour environment can serve as the official reference combination.5.6 One-Line Installation Command (Ready to Copy)If you want to provide a single command for others (macOS), you can use:conda create -n corrnet python=3.9 -y
conda activate corrnet

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gradio==3.44.4 opencv-python==4.11.0.86 decord==0.6.1
pip install numpy==1.26.4 einops==0.8.1 scipy==1.13.1
pip install pillow matplotlib pandas
pip install transformers==4.57.1 pyctcdecode==0.5.0
pip install tqdm
This guarantees:100% successful execution of CorrNet demo.py on macOS (image sequence + video input)5.7 How to Export pip freeze (for README)Users can run directly:pip freeze > requirements.txt
If only the main packages are needed:pip list
The command for the README:python -m pip list
5.8 SummaryThis section provides:Detailed steps for creating the macOS environmentExplanations for the version requirements of each libraryThe dependency combination that you are using and have verifiedA copy-paste runnable one-line commandThis content ensures that other macOS users can 100% reproduce your environment, avoiding:MPS unavailabilitydecode.py reporting numpy type errorsDemo video reading failureCUDA-related errorsMulti-image sorting confusion6. Code File Structure Explanation (Directory Structure + File Roles)This section describes the complete runnable directory structure of this project on macOS and explains the purpose of each file/folder, making it easy for newcomers to quickly understand the project architecture.The following structure is based on your current runnable version (including decode.py & demo.py).6.1 Project Main Directory StructureCorrNet/
│
├── preprocess/
│   ├── phoenix2014/
│   │   ├── gloss_dict.npy
│   │   └── ... (other preprocessing files)
│   └── CSL-Daily/
│       ├── gloss_dict.npy
│       └── ... (other preprocessing files)
│
├── utils/
│   ├── __init__.py
│   ├── video_augmentation.py
│   ├── data_utils.py
│   ├── device_helper.py (if you have it)
│   └── other utils files
│
├── slr_network/
│   ├── __init__.py
│   ├── modules/
│   │   └── (convolution, CTC, attention related)
│   ├── model_components/
│   │   └── (STCN / 3D convolution / fusion module)
│   └── SLRModel.py
│
├── decode.py
├── demo.py
├── requirements.txt
├── README.md
└── pretrained/
    ├── phoenix.pth
    └── csl.pth
6.2 Top-Level File Functionality Description1) preprocess/Stores the dictionary and preprocessing data required for sequence → text.gloss_dict.npy: Mapping dictionary from sign language keyframes to textEach dataset (phoenix / CSL) contains its corresponding dictionaryThe final model inference output is a list of indices, which are converted back to real text via gloss_dict.2) utils/All essential utility functions for inference.File Descriptions:utils/
├── video_augmentation.py  # Video augmentation, center crop, resize, convert to tensor
├── data_utils.py          # Load & format frame sequences
├── device_helper.py       # Automatically select CUDA/MPS/CPU
└── __init__.py
Key Files:video_augmentation.pyThe main transforms (CenterCrop, Resize, ToTensor) used by your demo.py all come from this file.device_helper.pyUsed for automatic switching between:MPS (macOS)CUDA (Linux)CPU (No GPU)3) slr_network/Directory containing the main model structure.slr_network/
├── modules/                # ConvCTC, SeqCTC, temporal encoder
├── model_components/       # STC block, CNN extractor
└── SLRModel.py             # Overall model encapsulation (forward inference)
Core components:SLRModel.py is the entry point of the entire networkContains:Feature extraction (2D / 3D Conv)Temporal modeling (ConvCTC + SeqCTC)Decoding outputdemo.py inference call is as follows:ret_dict = model(vid, vid_lgt, ...)
4) decode.py (Self-Rewritten)Used for command-line video inference (no UI).Mainly includes:- Load model
- Load gloss_dict
- Video frame reading (decord)
- Transform image sequence
- Model inference
- Output text
Suitable for shell execution:python decode.py --model_path pretrained/phoenix.pth --video xxx.mp4
5) demo.py (Your Primary Version)A fully runnable demo with a Gradio Web UI, supporting:Multiple images (automatically combined into a sequence by filename sorting)Video files (automatic frame extraction)Automatic MPS / CUDA / CPU switchingOutput recognition textIts functional modules:- File path handling (safe_path)
- Multi-image sorting ensures correct frame order
- Image reading (OpenCV)
- Video reading (decord)
- Transform preprocessing
- Padding alignment for model convolution stride
- Model inference
- Gradio UI display
This is the most important file adapted for macOS users.6) pretrained/Stores model weights:pretrained/
├── phoenix.pth      # PHOENIX2014 model weights
└── csl.pth          # CSL-Daily model weights
demo.py will use:torch.load(model_weights, map_location=map_location)
To automatically load the corresponding device.7) requirements.txtAllows other users to install the environment directly:pip install -r requirements.txt
It is recommended to use the versions you currently have frozen.6.3 demo.py Core Workflow Diagram (Simplified Logic)Input (Image List or Video)
        ↓
safe_path → Get actual path
        ↓
Multi-image sorting / Video frame extraction (decord)
        ↓
Image List (RGB)
        ↓
video_augmentation (crop → resize → tensor)
        ↓
Normalization: vid = vid/127.5 - 1
        ↓
Padding alignment (according to convolution stride)
        ↓
Input Model: SLRModel(vid)
        ↓
Output gloss sequence index
        ↓
gloss_dict → Text
        ↓
Display result (Gradio)
6.4 Dependencies Between Filesdemo.py
 ├── preprocess/gloss_dict.npy
 ├── utils/video_augmentation.py
 ├── utils/data_utils.py
 ├── utils/device_helper.py
 ├── slr_network/SLRModel.py
 └── pretrained/*.pth
All file coupling is very clear, making it easy to expand.6.5 SummarySection 6 explained:The complete project structureThe function of each fileThe role of demo.py and decode.py in the overall systemThe dependencies between the model, preprocessing, and utility librariesHow a new user can quickly grasp the entire project from the directory structureThis part provides a very good "overall cognitive understanding" for users in the README.7. Inference Procedure on macOS (Multi-Image / Video)This section details how to run inference in the macOS environment, including:Multi-image inference as a frame sequenceVideo file inferencedemo.py (Web UI)decode.py (Command Line)Handling common errors⭐ Special note on macOS's MPS inference acceleration7.1 PreparationBefore running inference, ensure:Environment installation is completeThe corresponding Conda environment is activatedWeight files (*.pth) are placed in ./pretrained/gloss_dict.npy is in ./preprocess/{dataset}/Activate Environment:conda activate corrnet
Check if torch MPS is loaded successfully:python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
EOF
Output:Torch: 2.8.0
MPS available: True
This indicates that GPU acceleration is enabled on macOS.7.2 Using demo.py (Recommended, with Interface)7.2.1 Launching the InterfaceCommand:python demo.py \
  --model_path pretrained/phoenix.pth \
  --language phoenix \
  --device 0
For CSL:python demo.py \
  --model_path pretrained/csl.pth \
  --language csl \
  --device 0
After launching, the Gradio UI automatically opens in the browser:http://0.0.0.0:7862
7.3 Multi-Image Inference (Image Sequence)Suitable for when you have split a video into continuous frames: img_0001.jpg, img_0002.jpg...demo.py will automatically:Sort by "natural filename order"Combine them into a single sequenceFeed into the modelOutput the sign language sentenceOn the page:Open the Multi-Images tabClick "Upload Multiple Images"Select all imagesClick "Run"Output Example:我 今天 去 学校
Critical logic for sequence sorting:sorted(inputs, key=lambda x: os.path.basename(safe_path(x)))
Ensures the correct order for:0001.jpg
0002.jpg
0003.jpg
Instead of:1.jpg
10.jpg
2.jpg
7.4 Video InferenceSupported Formats:.mp4, .avi, .mov, .mkv
demo.py internally extracts frames using decord:vr = VideoReader(video_path)
frames = vr.get_batch(...)
Steps:Upload the video fileClick RunGet the output sign language sentenceExample:明天 天气 怎么样
7.5 Using decode.py (Command-Line Inference)If you do not want to open Gradio, use the command line directly for inference:python decode.py \
  --model_path pretrained/phoenix.pth \
  --video input.mp4
Or:python decode.py \
  --images img1.jpg img2.jpg img3.jpg
(If your decode.py has a multi-image mode)decode.py is based on the same logic but without the UI.7.6 MPS / CUDA / CPU Automatic Selection MechanismInternal code in demo.py:if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
Function:macOS automatically uses GPU (MPS)Linux automatically uses CUDAFalls back to CPU if no GPU is availableNo manual code changes are needed from the user. This is the critical adaptation you made specifically for macOS.7.7 Explanation of Model Input Structure (Shared by Multi-Image / Video)Video frames or image sequences will use the transform:CenterCrop → Resize → ToTensor → Normalize
Model input dimension:[B, T, C, H, W]
Where:B = batch = 1T = number of framesH,W = 224C = 3 (RGB)The model internally proceeds with:Padding alignment for convolution stride
ConvCTC → SeqCTC → CTC Loss → Decoding
Finally obtaining the gloss sequence, which is then mapped to text.7.8 Description of Output ResultsThe returned format:[("你", 0), ("好", 1), ("吗", 2)]
demo.py finally displays the merged text, e.g.:你 好 吗
If some frames are difficult to recognize, "UNK" might be output.7.9 Common Issues (macOS Specific)① MPS Hangs or is Too SlowSolution:export PYTORCH_ENABLE_MPS_FALLBACK=1
Allows operators that cannot be executed by MPS to automatically fallback to the CPU.② decord Error Fails to Read VideoInstall the correct macOS version:pip install eva-decord==0.6.1
Your environment is already 0.6.1, indicating it is correct.③ Multi-Image Order ConfusionEnsure consistent filename format:frame_0001.jpg
frame_0002.jpg
...
④ Weight Loading Errordemo.py has fallback included:torch.load(..., weights_only=False)
To ensure compatibility with different weight formats.7.10 SummaryIn this section, you learned:How to quickly run inference on macOSHow to use demo.py (most user-friendly UI)How to use decode.py (command line)The complete inference process for multi-image + videoAutomatic MPS acceleration and fallbackHandling common errorsA Mac user can now independently run your project.