#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from lm_eval import evaluator, tasks
from lm_eval.models.dummy import DummyLM


def test_evaluate_dummy_model():
    """
    Check if lm_eval dummy model can get some results
    """
    res = evaluator.evaluate(DummyLM(), tasks.get_task_dict(["lambada"]), limit=2, write_out=False, cache_requests=True)

    assert "results" in res
    assert res["results"] is not None


if __name__ == "__main__":
    test_evaluate_dummy_model()
