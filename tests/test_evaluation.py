from evaluation.metrics import normalize_text, evaluate_required_facts, evaluate_out_of_scope

def test_normalize_text():
    text = "Board Diversity: 30% Women!"

    result = normalize_text(text)
    assert result == "board diversity 30% women"

def test_required_facts_pass():
    answer = ("The Board Diversity Policy aims for at least 30% representation by women.")
    required_facts = ["30%", "women"]

    passed, score, matched, missing = evaluate_required_facts(answer, required_facts)
    assert passed is True
    assert score == 1.0
    assert len(missing) == 0


def test_required_facts_fail():
    answer = "The policy discusses board diversity."
    required_facts = ["30%", "women"]

    passed, score, matched, missing = evaluate_required_facts(answer, required_facts)
    assert passed is False
    assert len(missing) > 0

def test_out_of_scope_detection():
    answer = "I couldn't find the answer in the provided documents."
    assert evaluate_out_of_scope(answer) is True

def test_normal_answer_not_out_of_scope():
    answer = "The policy applies to all directors and employees."
    assert evaluate_out_of_scope(answer) is False