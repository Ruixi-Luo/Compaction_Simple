"""官方 selective-context 0.1.4 的薄封装：只替换固定版本的模型加载位置。"""
from . import validate_input
from .runtime import model_directory, prepare_torch
from .chunking import split_overlong


def length_safe_sentences(sents, tokenizer, limit):
    count = lambda text: len(tokenizer.encode('<|endoftext|>' + text.strip(), add_special_tokens=False))
    return [part.strip() for sent in sents for part in split_overlong(sent, count, limit) if part.strip()]


class SelectiveContextCompressor:
    def __init__(self, device='cpu'):
        prepare_torch(device)
        from selective_context import SelectiveContext
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        folder = str(model_directory("selective_context"))

        class PinnedSelectiveContext(SelectiveContext):
            def _prepare_model(self):
                # 保留官方句内 self-information、短语聚合与百分位筛选代码。
                # 覆盖此钩子，只为指定设备、safetensors、离线固定模型路径。
                self.device = device
                self.tokenizer = GPT2Tokenizer.from_pretrained(folder, local_files_only=True)
                self.model = GPT2LMHeadModel.from_pretrained(
                    folder, local_files_only=True, use_safetensors=True
                ).to(self.device).eval()
                self.max_token_length = self.model.config.n_positions
                self.get_self_information = self._get_self_info_via_gpt2

            def _lexical_unit(self, sents):
                # 仅对超过位置上限的长句分段；全文仍共用官方短语筛选阈值。
                safe = length_safe_sentences(sents, self.tokenizer, self.max_token_length)
                return super()._lexical_unit(safe)

        self.engine = PinnedSelectiveContext(model_type="gpt2", lang="en")

    def compress(self, context, retained_ratio=0.5):
        """仅输入上下文和保留率，输出压缩文本；不接收 query。"""
        validate_input(context, retained_ratio)
        if not context.strip() or retained_ratio == 1:
            return context
        # 注意：官方参数表示删去比例；本项目统一参数表示保留比例。
        text, _removed_phrases = self.engine(
            context, reduce_ratio=1 - retained_ratio, reduce_level="phrase"
        )
        return text
