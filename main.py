# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("AntibodyGeneration/fine-tuned-progen2-large")