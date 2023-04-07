from langchain.llms import OpenAI
import os
from keys import openAI_API_KEY
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


os.environ["OPENAI_API_KEY"] = openAI_API_KEY
llm = OpenAI(temperature=0.8)

# simple use of the langchain functionality
# the llm (large language model) is key of communication in here
# we are sending to OpenAI the prompt by using the llm

text = "what are 5 vacation destinations for someone who likes to eat sushi"
print(llm(text))



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

# chains #
chain = LLMChain(llm=llm,prompt=prompt)
print(chain.run("fruit"))



