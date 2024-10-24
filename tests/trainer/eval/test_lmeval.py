from lm_eval import evaluator, tasks
from lm_eval.models.dummy import DummyLM


def test_evaluate_dummy_model():
    """
    Check if lm_eval dummy model can get some results
    """
    res = evaluator.evaluate(DummyLM(), tasks.get_task_dict(["lambada"]), limit=10, write_out=False)

    assert "results" in res
    assert res["results"] is not None


if __name__ == "__main__":
    test_evaluate_dummy_model()
