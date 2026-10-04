import ollama


def generate_response(prompt, model, endpoint=None, temperature=0, timeout=300):

    client = ollama.Client(host=endpoint, timeout=timeout)

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": temperature}
    )

    return response["message"]["content"]
