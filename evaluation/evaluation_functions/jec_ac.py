import logging
from utils.function_utils import multi_choice_judge

"""
Task: multi-choice selection
Metric: Accuracy
司法考试-案例分析
"""

_logger = logging.getLogger(__name__)


def compute_jec_ac(data_dict):
    """
    Compute the Accuracy
    The JEC dataset has 4 options for each question: A, B, C, D
    A prediction is correct if
    1. The correct answer appears in the prediction, and
    2. Options other than the answer do not appear in the prediction.
    """
    score_list, abstentions = [], 0
    option_list = ["A", "B", "C", "D"]
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        if not (answer.startswith("正确答案:") and len(answer) > 5 and answer[5] in option_list):
            _logger.warning("Skipping malformed answer: %r", answer)
            continue

        answer_letter = answer[5]
        judge = multi_choice_judge(prediction, option_list, answer_letter)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    if not score_list:
        return {"score": 0.0, "abstention_rate": 1.0}
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions / len(data_dict)}
