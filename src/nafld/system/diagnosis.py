from runscripts.manage_data.configs.step_1_5_config import THRESHOLD


def check_if_patient_is_healthy(prediction: float) -> float:
    if prediction > THRESHOLD:
        return round(prediction, 2), "chory"
    return round((1 - prediction), 2), "zdrowy"
