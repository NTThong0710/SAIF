from detoxify import Detoxify

# Load model một lần
detox_model = Detoxify('original')

# Hàm kiểm tra prompt
def is_prompt_safe(prompt: str):
    results = detox_model.predict(prompt)
    
    # Nếu bất kỳ chỉ số độc hại nào > 0.5 thì xem là không an toàn
    threshold = 0.5
    flagged = {label: score for label, score in results.items() if score > threshold}
    
    if flagged:
        return False, list(flagged.keys())
    return True, []
