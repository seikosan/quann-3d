from cloudlm.tokenizer import BPETokenizer, Tok


def test_bpe_encode_appends_eos(tmp_path):
    tokenizer = BPETokenizer(vocab_size=32).train(
        ["merhaba dünya", "merhaba dünya", "merhaba kitap"]
    )
    encoded = tokenizer.encode("merhaba dünya")
    assert encoded[-1] == tokenizer._tokenizer.token_to_id("<eos>")

    path = tmp_path / "bpe.json"
    tokenizer.save(str(path))
    loaded = BPETokenizer.load(str(path))
    assert loaded.encode("merhaba dünya")[-1] == loaded._tokenizer.token_to_id("<eos>")


def test_tok_stoi_is_callable_and_normalization_round_trips(tmp_path):
    tokenizer = Tok(punct=True, fold_i=True)
    tokenizer.build(["IĞDIR, güzel."])
    assert tokenizer.stoi("bilinmeyen") == 2
    assert tokenizer.stoi.get("<eos>") == 1

    path = tmp_path / "tok.json"
    tokenizer.save(str(path))
    loaded = Tok.load(str(path))
    assert loaded.punct is True
    assert loaded.fold_i is True
    assert loaded.encode("IĞDIR, güzel.") == tokenizer.encode("IĞDIR, güzel.")
