from pathlib import Path
from src.model import model, HParams
from src.encoder import get_encoder


if __name__ == '__main__':
    txt_path = Path("/media/gael/Space/PycharmProjects/gpt-2/contemplations_clean.txt").resolve()
    poems = open(txt_path, "r", encoding='utf-8-sig').read()
    poems = poems.lower().replace("\n", " ").split(' ')
    hparams = HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12
    )
    model = model(hparams, poems)
    pass
