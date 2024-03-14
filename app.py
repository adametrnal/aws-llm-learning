import boto3
import json
import os
import uuid
import time

from IPython.display import Audio
from jinja2 import Template


#Boto Clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name="us-east-1")
s3_client = boto3.client('s3', region_name='us-east-1')
transcribe_client = boto3.client('transcribe', region_name='us-east-1')

# Basic Prompting using Bedrock
# prompt = "How do I build a house?"

# kwargs = {
#     "modelId": "amazon.titan-text-express-v1",
#     "contentType": "application/json",
#     "accept": "*/*",
#     "body": json.dumps(
#         {
#             "inputText": prompt,
#             "textGenerationConfig": {
#                 "maxTokenCount": 500,
#                 "temperature": 0.7,
#                 "topP": 0.9
#             }
#         }
#     )
# }

# response = bedrock_runtime.invoke_model(**kwargs)

# print(response)
# response_body = json.loads(response.get('body').read())
# print(json.dumps(response_body, indent=4))
# print(response_body['results'][0]['outputText'])


# Now let's try importing an Audio file 
file_name = "adam_dialog.mp3"
# os.system("afplay " + file_name)
bucket_name = "brill-bucket"

s3_client.upload_file("../Docs/" + file_name, bucket_name, file_name)
job_name = 'transcription-job-' + str(uuid.uuid4())

response = transcribe_client.start_transcription_job(
    TranscriptionJobName=job_name,
    Media={'MediaFileUri': f's3://{bucket_name}/{file_name}'},
    MediaFormat='mp3',
    LanguageCode='en-US',
    OutputBucketName=bucket_name,
    Settings={
        'ShowSpeakerLabels': True,
        'MaxSpeakerLabels': 2
    }
)

# Simple poll to check until it's done 
while True:
    status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    time.sleep(2)
print(status['TranscriptionJob']['TranscriptionJobStatus'])


if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
    
    # Load the transcript from S3.
    transcript_key = f"{job_name}.json"
    transcript_obj = s3_client.get_object(Bucket=bucket_name, Key=transcript_key)
    transcript_text = transcript_obj['Body'].read().decode('utf-8')
    transcript_json = json.loads(transcript_text)
    
    output_text = ""
    current_speaker = None
    
    items = transcript_json['results']['items']
    
    for item in items:
        
        speaker_label = item.get('speaker_label', None)
        content = item['alternatives'][0]['content']
        
        # Start the line with the speaker label:
        if speaker_label is not None and speaker_label != current_speaker:
            current_speaker = speaker_label
            output_text += f"\n{current_speaker}: "
            
        # Add the speech content:
        if item['type'] == 'punctuation':
            output_text = output_text.rstrip()
            
        output_text += f"{content} "
        
    # Save the transcript to a text file
    with open(f'./transcripts/{job_name}.txt', 'w') as f:
        f.write(output_text)


# Let's summarize the content with an LLM
with open(f'./transcripts/{job_name}.txt', "r") as file:
    transcript = file.read()

#Load in our template from an external template for separation of concerns
with open('prompt_template.txt', "r") as file:
    template_string = file.read()

print("template string:")
print(template_string) 

data = {
    'transcript' : transcript
}
template = Template(template_string)
prompt = template.render(data)

print("prompt")
print(prompt)

# Now use an LLM to summarize
kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 500,
                "temperature": 0,
                "topP": 0.9
            }
        }
    )
}
response = bedrock_runtime.invoke_model(**kwargs)

print(response)
response_body = json.loads(response.get('body').read())
print(json.dumps(response_body, indent=4))
print(response_body['results'][0]['outputText'])
