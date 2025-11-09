import sys
import os
import json
import re
from functools import lru_cache
import jieba
from funasr import AutoModel

def normalize_text(text: str) -> str:
    """(已修改) 仅去除首尾空格，保留FunASR输出的标点符号。"""
    # 保留原始文本，仅去除可能存在于开头或结尾的空白
    return text.strip()

@lru_cache(maxsize=1)
# FunASR 推荐使用 paraformer-zh 模型，它在中文上表现优异
def get_model(device, model_name="paraformer-zh", use_vad=False):
    """
    缓存 FunASR 模型到全局。
    模型在第一次使用时会自动从 ModelScope 下载。
    """
    # 设置模型缓存路径到项目根目录下的 local_model 文件夹
    # os.path.dirname(__file__) 获取当前文件所在目录的绝对路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(project_dir, "local_model")
    os.environ['MODELSCOPE_CACHE'] = cache_dir
    
    model_kwargs = {
        "model": model_name,
        "punc_model": "ct-punc",
        "device": device
    }
    if use_vad:
        model_kwargs["vad_model"] = "fsmn-vad"
    
    model = AutoModel(**model_kwargs)
    return model

def transcribe_audio(wav_path, asr_config):
    """
    对音频执行 ASR，返回带时间戳的分段 (词级别)。
    """
    use_gpu = asr_config.get('use_gpu', True)
    device = "cuda:0" if use_gpu else "cpu"
    
    funasr_config = asr_config.get('funasr', {})
    funasr_model_name = funasr_config.get('model', 'paraformer-zh')
    use_vad = funasr_config.get('use_vad', True)

    model = get_model(device, funasr_model_name, use_vad)
    
    res = model.generate(wav_path, batch_size_s=300)

    if not (res and isinstance(res, list) and "text" in res[0]):
        return []

    # 1. 清理空格并获取逐字时间戳
    full_text = res[0]["text"].replace(" ", "")
    char_timestamps = res[0].get("timestamp", [])
    
    if not full_text or not char_timestamps:
        return []

    # 2. 使用 jieba 进行分词
    words = list(jieba.cut(full_text))

    # 3. 计算每个词语的时间戳
    result = []
    char_index = 0
    for word in words:
        if not word.strip():  # 跳过空格等无效词
            continue
        
        word_len = len(word)
        if char_index + word_len > len(char_timestamps):
            # 如果分词结果和时间戳对不上，就此打住
            break

        # 词的开始时间 = 第一个字的开始时间
        start_ms = char_timestamps[char_index][0]
        # 词的结束时间 = 最后一个字的结束时间
        end_ms = char_timestamps[char_index + word_len - 1][1]
        
        result.append({
            "start": float(start_ms) / 1000.0,
            "end": float(end_ms) / 1000.0,
            "text": word
        })
        
        char_index += word_len
        
    return result

if __name__ == "__main__":
    # 主程序入口仅用于测试，可以简化或按需修改
    if len(sys.argv) < 3:
        print("Usage: python asr.py <wav_path> <output_json>")
        sys.exit(1)

    wav_path = sys.argv[1]
    output_json = sys.argv[2]
    
    # 模拟 asr_config
    asr_config_for_test = {'use_gpu': True}

    try:
        # model_dir 参数已被移除
        result = transcribe_audio(wav_path, asr_config_for_test)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"error": str(e)}, f, ensure_ascii=False)
        sys.exit(1)