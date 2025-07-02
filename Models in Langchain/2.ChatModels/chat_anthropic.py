from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

model=ChatAnthropic(model='model_name',temperature=0.6,max_completion_tokens=250)
rslt=model.invoke("Whats going on ??")
print(rslt)
