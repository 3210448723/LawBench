import re
import logging

"""
number prediction
metric: accuracy
金额提取
"""

_logger = logging.getLogger(__name__)


def compute_jetq(data_dict):
    """
    Compute the Accuracy
    we extract the total amount of cost involved in the crime from the prediction and compare it with the reference
    The prediction is correct if
    the total amount of cost provided in the reference, appears in the prediction.
    """
    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        if not (answer.startswith("上文涉及到的犯罪金额:") and answer.endswith("元。")):
            _logger.warning("Skipping malformed answer: %r", answer)
            abstentions += 1
            score_list.append(0)
            continue
        answer_clean = answer.replace("上文涉及到的犯罪金额:", "")
        if "千元" in answer_clean or "万" in answer_clean:
            _logger.warning("Unexpected unit in answer %r; skipping.", answer)
            abstentions += 1
            score_list.append(0)
            continue

        # remove "元。"
        answer_clean = answer_clean.replace("元。", "")
        try:
            answer_val = float(answer_clean)
        except ValueError:
            _logger.warning("Cannot parse answer value %r; skipping.", answer_clean)
            abstentions += 1
            score_list.append(0)
            continue

        prediction_digits = re.findall(r"\d+\.?\d*", prediction)
        prediction_digits = [float(digit) for digit in prediction_digits]

        if len(prediction_digits) == 0:
            abstentions += 1
        if answer_val in prediction_digits:
            score_list.append(1)
        else:
            score_list.append(0)


    # compute the accuracy of score_list
    if not score_list:
        return {"score": 0.0, "abstention_rate": 1.0}
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions/len(data_dict)}
