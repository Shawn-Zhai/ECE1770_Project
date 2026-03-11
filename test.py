from openai import OpenAI

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello"}],
)

print(resp.choices[0].message.content)