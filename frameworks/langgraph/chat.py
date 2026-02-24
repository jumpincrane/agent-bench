import uuid

from langchain_core.messages import HumanMessage

from agent import graph


thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print(f"Sesja: {thread_id}")
print("Wpisz 'exit' żeby zakończyć, 'new' żeby zacząć nową rozmowę\n")

while True:
    user_input = input("Ty: ").strip()

    if user_input.lower() == "exit":
        break

    if user_input.lower() == "new":
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        print(f"Nowa sesja: {thread_id}\n")
        continue

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    print(f"Agent: {result['messages'][-1].content}\n")
