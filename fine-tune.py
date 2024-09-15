import openai
import json
from datetime import datetime
import logging
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Set your OpenAI API key
openai.api_key = config["OpenAI"]["open_ai_api"]

def prepare_training_data(data):
    training_data = []
    for faq in data["faqs"]:
        training_data.append({
            "messages": [
                {"role": "system", "content": data["initial_prompt"]},
                {"role": "user", "content": faq["question"]},
                {"role": "assistant", "content": faq["answer"]}
            ]
        })
    return training_data

if __name__ == "__main__":
    
    json_data = r"data\master_data.json"
    json_data_job = r"fine-tuned\training_data.jsonl"

    # Prepare the training data
    with open(json_data, "r", encoding="utf-8") as f:
        data = json.load(f)

    training_data = prepare_training_data(data)

    # Save the training data to a JSONL file
    with open(json_data_job, "w", encoding="utf-8") as f:
        for entry in training_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # # Create a fine-tuning job
    # try:
    #     response = openai.File.create(
    #         file=open(json_data_job, "rb"),
    #         purpose='fine-tune'
    #     )
    #     file_id = response.id
    #     logging.info(f"File uploaded successfully. File ID: {file_id}")

    #     # Create the fine-tuning job
    #     job = openai.FineTuningJob.create(
    #         training_file=file_id,
    #         model="gpt-3.5-turbo"
    #     )
    #     openai.fine_tuning.jobs.create(
    #     training_file="file-abc123", 
    #     model="gpt-4o-mini"
    #     )
    #     logging.info(f"Fine-tuning job created successfully. Job ID: {job.id}")

    # except openai.error.OpenAIError as e:
    #     with open("error.log", "+a") as error_log:
    #         error_log.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {e}")
