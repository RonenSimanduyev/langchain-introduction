from langchain.llms import OpenAI
import os
from keys import openAI_API_KEY
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator




chat = ChatOpenAI(temperature=.7, openai_api_key=openAI_API_KEY)

os.environ["OPENAI_API_KEY"] = openAI_API_KEY
llm = OpenAI(temperature=0.8)

# @@@@@@ introduction ! @@@@@@@

'''
# simple use of the langchain functionality
# the llm (large language model) is key of communication in here
# we are sending to OpenAI the prompt by using the llm

text = "what are 5 vacation destinations for someone who likes to eat sushi"
print(llm(text))
'''


# @@@@@@ chains ! @@@@@@@
'''

# in here we are making a template of the prompt and changing the values that is the main
# part of the question
prompt= PromptTemplate(
    input_variables=["food"],
    template="what are 5 vacation destinations for someone who likes to eat {food} ?"
)

print(prompt.format(food="dessert"))
# result should be :
# what are 5 vacation destinations for someone who likes to eat dessert
print(llm(prompt.format(food="dessert")))



chain = LLMChain(llm=llm,prompt=prompt)
print(chain.run("fruit"))
'''











# @@@@@@ chats ! @@@@@@@


# in this example you can see that we can have pre conversation with as a chat so we could build him
# according to our requirements

# Chat Messages
# Like text, but specified with a message type (System, Human, AI)
#
# System - Helpful background context that tell the AI what to do
# Human - Messages that are intented to represent the user
# AI - Messages that show what the AI responded with
'''

messages = [
    SystemMessage(content="You are a nice AI bot that helps to learn new languages"),
    HumanMessage(content="what is the easiest language to learn")
]

response = chat(messages)
print(response.content)

# another example
# you can also change teh response to China or russia and he will tell you to what to do in those places
# even though that the expected answer is Japan
conversation = [
        SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
        HumanMessage(content="I like to eact sushi and watch anime where should I go?"),
        AIMessage(content="You should go to Nice, japan"),
        HumanMessage(content="What else should I do when I'm there?")
    ]


response = chat(conversation)
print(response.content)
'''
