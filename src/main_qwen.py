import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
import sys

# Toggle to switch between full response and extracted description
OUTPUT_FULL_RESPONSE = False

# Ensure we're using the MPS device if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

path_input = ""


def get_image_path():
    """Prompt user for an image path until a valid path is provided or 'q' to quit."""
    while True:
        path_input = input("\nEnter the path to your image or 'q' to quit: ").strip().strip("'\"")
        if path_input.lower() == 'q':
            return None
        print("path input print!!")
        return path_input
        # Path 오류로 아래 주석 -> 이미지 링크 입력하도록 변경
        # path = Path(path_input)
        # if path.exists():
        #     return path
        # print("The file does not exist. Please try again.")


def load_model_and_tools():
    """Load the Qwen2-VL model, tokenizer, and processor for Apple Silicon."""
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    try:
        # model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", offload_buffers=True).to(device)
        # iOS 용
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16,
                                                                device_map="auto", offload_buffers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, tokenizer, processor
    except Exception as e:
        print(f"Failed to load the model: {e}")
        sys.exit(1)


def process_image(image_path: path_input, model, tokenizer, processor):
    """Process the image and generate a description using the MPS device if available."""
    try:
        # with Image.open(image_path) as img:
        # prompt = "Describe this image in detail as prose and respond to this English sentence in Korean: Identify the type of image (photo, diagram, etc.). Describe each person if any, using specific terms like 'man', 'woman', 'boy', 'girl' etc. and include details about their attire and actions. Guess the location. Include sensory details and emotions. Don't be gender neutral. Avoid saying 'an individual', instead be specific about any people."
        prompt = "이미지 안에 있는 내용을 markdown 형태의 표로 만들어줘."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        # inputs = processor(text=[text], images=[image_path], padding=True, return_tensors="pt")

        # Generate outputs
        outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.5, repetition_penalty=1.1)

        # Decode the outputs using tokenizer
        response_ids = outputs[0]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=False)

        if OUTPUT_FULL_RESPONSE:
            print(f"\n{response_text}")
        else:
            response = response_text.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()
            print(f"\n{response}")
    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
    except Image.UnidentifiedImageError:
        print(f"Error: The file '{image_path}' is not a valid image file or is corrupted.")
    except torch.cuda.OutOfMemoryError:
        print("Error: Ran out of GPU memory. Try using a smaller image or freeing up GPU resources.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{image_path}': {e}")
        print("Please check your input and try again.")


def main():
    """Main function to handle model loading and image processing loop."""
    print(f"Loading model on {device.type} device...")
    model, tokenizer, processor = load_model_and_tools()

    image_path = '/Users/glee/Desktop/sakak/TSA/samples/1. 진단서 샘플.jpg'

    # while True:
    #     image_path = get_image_path()
    #     if image_path is None:
    #         print("Exiting...")
    #         break
    process_image(image_path, model, tokenizer, processor)

    print("Goodbye!")


if __name__ == "__main__":
    main()