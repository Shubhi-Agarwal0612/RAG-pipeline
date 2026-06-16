from main import rewrite, query

doc_ids = [8]   # the test PDF's id in your DB — use the same id that worked in eval.py

history = []
while True:
    question = input("\nQuestion (or 'new' to reset): ")

    if question == 'new':
        history = []
        print("--- conversation reset ---")
        continue

    standalone_question = rewrite(history, question)
    print(f"[rewritten: {standalone_question}]")          # so you can SEE the rewrite

    answer, chunks = query(standalone_question, doc_ids)
    print(f"\nAnswer: {answer}")

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    history = history[-10:]