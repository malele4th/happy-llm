from config import create_agent

if __name__ == "__main__":
    agent = create_agent()

    while True:
        prompt = input("\033[94mUser: \033[0m")
        if prompt == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)
