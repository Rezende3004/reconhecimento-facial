from reconhecimento_facial.labels import LabelStore


def test_cria_e_recupera_label(tmp_path):
    labels_path = tmp_path / "labels.json"
    store = LabelStore(labels_path)

    label_id = store.get_or_create_id("Joao")
    same_label_id = store.get_or_create_id("Joao")

    assert label_id == same_label_id
    assert store.get_name(label_id) == "Joao"


def test_normaliza_nome(tmp_path):
    labels_path = tmp_path / "labels.json"
    store = LabelStore(labels_path)

    label_id = store.get_or_create_id("  Maria   Silva  ")

    assert store.get_name(label_id) == "Maria Silva"
