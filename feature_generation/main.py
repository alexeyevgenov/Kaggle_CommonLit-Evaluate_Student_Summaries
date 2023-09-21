from data_processing_unit import Preprocessor, preprocess_and_join
import pandas as pd
from config import CONFIG
from multiprocessing import Process, Queue, Pool
from concurrent.futures import ThreadPoolExecutor
import time


TEST_MODE = True


def data_preparation() -> None:
    # DATA_DIR = f"../data/"
    # prompt_grade = pd.read_csv(f"{DATA_DIR}prompt_grade.csv")
    # prompts_train = pd.read_csv(DATA_DIR + "prompts_train.csv")
    # prompts_test = pd.read_csv(DATA_DIR + "prompts_test.csv")
    # if TEST_MODE:
    #     summaries_train = pd.read_csv(DATA_DIR + "summaries_train.csv")[:50]
    # else:
    #     summaries_train = pd.read_csv(DATA_DIR + "summaries_train.csv")
    # summaries_test = pd.read_csv(DATA_DIR + "summaries_test.csv")
    # prompts_train = preprocess_and_join(prompts_train, prompt_grade, 'prompt_title', 'title', 'grade')
    # prompts_test = preprocess_and_join(prompts_test, prompt_grade, 'prompt_title', 'title', 'grade')
    #
    # data_preprocessor = Preprocessor(test_mode=TEST_MODE)
    # train = data_preprocessor.run_2(prompts_train, summaries_train)
    # test = data_preprocessor.run_2(prompts_test, summaries_test)

    data_preprocessor = Preprocessor(test_mode=TEST_MODE)
    for i in range(CONFIG.num_folds):
        df_processed = pd.DataFrame()
        print(f"\nPREPROCESSING THE FOLD {i}:")
        if TEST_MODE:
            input_df = pd.read_feather(path=CONFIG.storage + f"/fold {i}.ftr")[:51]
        else:
            input_df = pd.read_feather(path=CONFIG.storage + f"/fold {i}.ftr")

        # Divide the DataFrame into smaller chunks
        N_CHUNKS = 2
        chunk_size = input_df.shape[0] // N_CHUNKS
        chunks = [input_df[i:i + chunk_size] for i in range(0, len(input_df), chunk_size)]
        if len(chunks) > N_CHUNKS:
            chunks_copy = chunks.copy()
            chunks_copy = chunks_copy[:-2]
            chunks_copy.append(pd.concat(chunks[-2:]))
            chunks = [el.reset_index(drop=True) for el in chunks_copy]
            del chunks_copy

        # pool = Pool()
        # results = pool.map(data_preprocessor.run_pandas, chunks)
        # df_processed = pd.concat(results)
        # df_processed.to_feather(CONFIG.storage + f"/preprocessed fold {i}.ftr")

        processes = []
        queues = []
        for chunk in chunks:
            queue = Queue()
            queues.append(queue)
            try:
                p = Process(target=data_preprocessor.run_pandas, args=(chunk, queue))
            except Exception as ex:
                print(f"Error in process. {ex}")
            processes.append(p)
            p.start()

        for queue in queues:
            if df_processed.empty:
                df_processed = queue.get()
            else:
                df_processed = pd.concat([df_processed, queue.get()], axis=0)

        for p in processes:
            p.join()

        df_processed.reset_index(drop=True).to_feather(CONFIG.storage + f"/preprocessed fold {i}.ftr")


if __name__ == "__main__":
    start = time.time()
    data_preparation()
    duration = time.time() - start
    print(f"Time : {duration}")
