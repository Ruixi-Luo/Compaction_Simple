"""官方 llmlingua 0.2.2 的 LLMLingua-2 路径；不调用 LLMLingua-1。"""
from . import validate_input
from .runtime import model_directory, prepare_torch
from .chunking import split_overlong


class LLMLingua2Compressor:
    def __init__(self, device='cpu'):
        torch = prepare_torch(device)
        from llmlingua import PromptCompressor
        class LengthSafePromptCompressor(PromptCompressor):
            def _PromptCompressor__chunk_context(self, origin_text, chunk_end_tokens):
                chunks = super()._PromptCompressor__chunk_context(origin_text, chunk_end_tokens)
                # 原生分块重新分词可能溢出。最多 511 正文 token + CLS；不丢正文。
                count = lambda text: len(self.tokenizer.tokenize(text))
                return [part for chunk in chunks for part in split_overlong(chunk, count, 511)]

        self.engine = LengthSafePromptCompressor(
            model_name=str(model_directory("llmlingua2")), device_map=device,
            model_config={"local_files_only": True, "trust_remote_code": False,
                          "torch_dtype": torch.float32},
            use_llmlingua2=True, llmlingua2_config={"max_batch_size": 1},
        )
        self.engine.model.eval()

    def compress(self, context, retained_ratio=0.5):
        """仅输入上下文和保留率，输出压缩文本；不接收 query。"""
        validate_input(context, retained_ratio)
        if not context.strip() or retained_ratio == 1:
            return context
        # 直接调用二代专属接口。此接口没有 question / instruction 参数。
        result = self.engine.compress_prompt_llmlingua2(
            context=[context], rate=retained_ratio, target_token=-1,
            use_context_level_filter=False, use_token_level_filter=True,
            token_to_word="mean", force_tokens=[], force_reserve_digit=False,
            drop_consecutive=False,
        )
        return result["compressed_prompt"]
