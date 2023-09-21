import re
from collections import Counter
from typing import List
import numpy as np
import pandas as pd
# import modin.pandas as pd
# import swifter
import pyphen
import spacy
from autocorrect import Speller
from nltk.corpus import stopwords
from nltk import sent_tokenize, pos_tag
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from pandas import DataFrame
from scipy.stats import entropy as scipy_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from spellchecker import SpellChecker
import ast
from tqdm import tqdm
from config import CONFIG
from multiprocessing import Queue
import multiprocessing

tqdm.pandas()


N_ROWS = 50
features_for_norm = ["summary_length", "splling_err_num", "word_overlap_count", "bigram_overlap_count",
                     "trigram_overlap_count", "quotes_count"]
drop_columns = ["prompt_length", "prompt_id", "prompt_question", "prompt_title", "prompt_text", "student_id", "text",
                "full_text", "fixed_summary_text", "embeddings", "fold"]


# nltk.download('stopwords')  # todo: must to be downloaded
# nltk.download('punkt')  # todo: must to be downloaded
# python - m spacy download en_core_web_sm  # todo: must to be downloaded
# nltk.download('averaged_perceptron_tagger')  # todo: must to be downloaded


class Preprocessor:
    def __init__(self, test_mode: bool) -> None:
        self.test_mode = test_mode
        self.STOP_WORDS = set(stopwords.words('english'))
        self.spacy_ner_model = spacy.load('en_core_web_sm')
        self.speller = Speller(lang='en')
        self.spellchecker = SpellChecker()
        self.scaler = StandardScaler()

    @staticmethod
    def calculate_text_similarity(row):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([row['prompt_text'], row['text']])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]

    @staticmethod
    def sentiment_analysis(text):
        analysis = TextBlob(text)   # .sentiment
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity

    @staticmethod
    def calculate_sentiment_vader(text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score

    @staticmethod
    def calculate_lexical_characteristics(text):
        words = word_tokenize(text.lower())

        # Рассчитываем частоту встречаемости каждого слова
        word_freq = FreqDist(words)
        # Преобразуем частоты в вероятности
        word_prob = [word_freq[word] / len(words) for word in word_freq.keys()]
        # Рассчитываем лексическую энтропию
        lexical_entropy = scipy_entropy(word_prob, base=2)
        # Рассчитываем разнообразие лексики
        lexical_diversity = len(word_freq) / len(words)

        return lexical_entropy, lexical_diversity

    def word_overlap_count(self, row):
        """ intersection(prompt_text, text) """

        def check_is_stop_word(word):
            return word in self.STOP_WORDS

        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))

    @staticmethod
    def ngrams(token, n):
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int) -> int:
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)
        return len(common_ngrams)

    @staticmethod
    def quotes_count(row):  # todo: it is good to add a function of checking for copy-pasting
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary) > 0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text):
        wordlist = text.split()
        amount_miss = len(list(self.spellchecker.unknown(wordlist)))
        return amount_miss

    @staticmethod
    def calculate_unique_words(text):
        unique_words = set(text.split())
        return len(unique_words)

    def add_spelling_dictionary(self, tokens: List[str]) -> None:  # List[str]
        """dictionary update for pyspell checker and autocorrect"""
        self.spellchecker.word_frequency.load_words(tokens)
        self.speller.nlp_data.update({token: 1000 for token in tokens})

    def calculate_pos_ratios(self, text):
        pos_tags = pos_tag(word_tokenize(text))
        pos_counts = Counter(tag for word, tag in pos_tags)
        total_words = len(pos_tags)
        ratios = {tag: count / total_words for tag, count in pos_counts.items()}
        return ratios

    def calculate_punctuation_ratios(self, text):
        total_chars = len(text)
        punctuation_counts = Counter(char for char in text if char in '.,!?;:"()[]{}')
        ratios = {char: count / total_chars for char, count in punctuation_counts.items()}
        return ratios

    @staticmethod
    def calculate_keyword_density(row):
        keywords = set(row['prompt_text'].split())
        text_words = row['text'].split()
        keyword_count = sum(1 for word in text_words if word in keywords)
        return keyword_count / len(text_words)

    @staticmethod
    def count_syllables(word):
        dic = pyphen.Pyphen(lang='en')
        hyphenated_word = dic.inserted(word)
        return len(hyphenated_word.split('-'))

    def flesch_reading_ease_manual(self, text):
        total_sentences = len(TextBlob(text).sentences)
        total_words = len(TextBlob(text).words)
        total_syllables = sum(self.count_syllables(word) for word in TextBlob(text).words)

        if total_sentences == 0 or total_words == 0:
            return 0

        flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        return flesch_score

    def flesch_kincaid_grade_level(self, text):
        total_sentences = len(TextBlob(text).sentences)
        total_words = len(TextBlob(text).words)
        total_syllables = sum(self.count_syllables(word) for word in TextBlob(text).words)

        if total_sentences == 0 or total_words == 0:
            return 0

        fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        return fk_grade

    def gunning_fog(self, text):
        total_sentences = len(TextBlob(text).sentences)
        total_words = len(TextBlob(text).words)
        complex_words = sum(1 for word in TextBlob(text).words if self.count_syllables(word) > 2)

        if total_sentences == 0 or total_words == 0:
            return 0

        fog_index = 0.4 * ((total_words / total_sentences) + 100 * (complex_words / total_words))
        return fog_index

    @staticmethod
    def calculate_sentiment_scores(text):
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        return sentiment_scores

    def count_difficult_words(self, text, syllable_threshold=3):
        words = TextBlob(text).words
        difficult_words_count = sum(1 for word in words if self.count_syllables(word) >= syllable_threshold)
        return difficult_words_count

    def run_swifter(self) -> None:
        import pandas as pd
        import swifter
        for i in range(CONFIG.num_folds):
            print(f"\nPREPROCESSING THE FOLD {i}:")
            if self.test_mode:
                input_df = pd.read_feather(path=CONFIG.storage + f"/fold {i}.ftr")[:N_ROWS]
            else:
                input_df = pd.read_feather(path=CONFIG.storage + f"/fold {i}.ftr")

            # lexical_entropy, lexical_diversity
            # tqdm.pandas(desc='lexical characteristics preparation')
            input_df['lexical_entropy'], input_df['lexical_diversity'] = zip(
                *input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_lexical_characteristics))

            # the length of a text
            input_df['text_len_symbols'] = input_df['text'].swifter.allow_dask_on_strings().apply(len)
            input_df['text_len_words'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda text: len(text.split()))
            input_df['text_len_sentences'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda text: len(sent_tokenize(text)))

            # the length of words and sentences
            input_df['sentence_length_in_words'] = input_df['text_len_words'] / input_df['text_len_sentences']
            input_df['word_length_in_symbols'] = input_df['text_len_symbols'] / input_df['text_len_words']

            # Sentiment analysis
            # tqdm.pandas(desc="sentiment analysis preparation")
            input_df['vader_sentiment_scores'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_sentiment_vader)
            input_df['vader_sentiment_positive'] = input_df['vader_sentiment_scores'].swifter.apply(lambda x: x['pos'])
            input_df['vader_sentiment_negative'] = input_df['vader_sentiment_scores'].swifter.apply(lambda x: x['neg'])
            input_df['vader_sentiment_neutral'] = input_df['vader_sentiment_scores'].swifter.apply(lambda x: x['neu'])
            input_df['vader_sentiment_compound'] = input_df['vader_sentiment_scores'].swifter.apply(lambda x: x['compound'])

            # check the level of subjectivity and polarity
            # input_df['textblob_sentiment_scores'] = input_df['text'].progress_apply(self.sentiment_analysis)
            # input_df['textblob_polarity'] = input_df['textblob_sentiment_scores'].apply(lambda x: x.polarity)
            # input_df['textblob_subjectivity'] = input_df['textblob_sentiment_scores'].apply(lambda x: x.subjectivity)

            # tqdm.pandas(desc="prompt_length preparation")
            input_df["prompt_length"] = input_df["prompt_text"].swifter.allow_dask_on_strings().apply(lambda x: len(word_tokenize(x)))
            input_df["prompt_tokens"] = input_df["prompt_text"].swifter.allow_dask_on_strings().apply(word_tokenize)

            # tqdm.pandas(desc="summary_length preparation")
            input_df["summary_length"] = input_df["text"].swifter.allow_dask_on_strings().apply(lambda x: len(word_tokenize(x)))
            input_df["summary_tokens"] = input_df["text"].swifter.allow_dask_on_strings().apply(word_tokenize)

            # Add prompt tokens into spelling checker dictionary
            # tqdm.pandas(desc="prompt_tokens preparation")
            input_df["prompt_tokens"].swifter.allow_dask_on_strings().apply(self.add_spelling_dictionary)

            # fix misspelling
            # tqdm.pandas(desc="fixed_summary_text preparation")
            input_df["fixed_summary_text"] = input_df["text"].swifter.allow_dask_on_strings().apply(self.speller)

            input_df['gunning_fog_prompt'] = input_df['prompt_text'].swifter.allow_dask_on_strings().apply(self.gunning_fog)
            input_df['flesch_kincaid_grade_level_prompt'] = input_df['prompt_text'].swifter.allow_dask_on_strings().apply(
                self.flesch_kincaid_grade_level)
            input_df['flesch_reading_ease_prompt'] = input_df['prompt_text'].swifter.allow_dask_on_strings().apply(self.flesch_reading_ease_manual)

            # count misspelling
            # tqdm.pandas(desc="splling_err_num preparation")
            input_df["splling_err_num"] = input_df["text"].swifter.allow_dask_on_strings().apply(self.spelling)

            input_df['flesch_reading_ease'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.flesch_reading_ease_manual)
            input_df['word_count'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: len(x.split()))
            input_df['sentence_length'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: len(x.split('.')))
            input_df['vocabulary_richness'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: len(set(x.split())))

            input_df['word_count2'] = [len(t.split(' ')) for t in input_df.text]
            input_df['num_unq_words'] = [len(list(set(x.lower().split(' ')))) for x in input_df.text]
            input_df['num_chars'] = [len(x) for x in input_df.text]

            # Additional features
            input_df['avg_word_length'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: np.mean([len(word) for word in x.split()]))
            input_df['comma_count'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: x.count(','))
            input_df['semicolon_count'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: x.count(';'))

            # after merge preprocess
            # tqdm.pandas(desc='length_ratio preparation')
            input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']

            # tqdm.pandas(desc='word_overlap_count preparation')
            input_df['word_overlap_count'] = input_df.swifter.apply(self.word_overlap_count, axis=1)

            # tqdm.pandas(desc='bigram_overlap_count preparation')
            input_df['bigram_overlap_count'] = input_df.swifter.apply(self.ngram_co_occurrence, args=(2,), axis=1)
            input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)

            # tqdm.pandas(desc='trigram_overlap_count preparation')
            input_df['trigram_overlap_count'] = input_df.swifter.apply(self.ngram_co_occurrence, args=(3,), axis=1)
            input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)

            # tqdm.pandas(desc='quotes_count preparation')
            input_df['quotes_count'] = input_df.swifter.apply(self.quotes_count, axis=1)

            input_df['question_count'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: x.count('?'))
            input_df['pos_ratios'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_pos_ratios)

            # Convert the dictionary of POS ratios into a single value (mean)
            input_df['pos_mean'] = input_df['pos_ratios'].swifter.apply(lambda x: np.mean(list(x.values())))
            input_df['punctuation_ratios'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_punctuation_ratios)

            # Convert the dictionary of punctuation ratios into a single value (sum)
            input_df['punctuation_sum'] = input_df['punctuation_ratios'].swifter.apply(lambda x: np.sum(list(x.values())))
            input_df['keyword_density'] = input_df.swifter.apply(self.calculate_keyword_density, axis=1)
            input_df['jaccard_similarity'] = input_df.swifter.allow_dask_on_strings().apply(
                lambda row: len(set(word_tokenize(row['prompt_text'])) & set(word_tokenize(row['text']))) / len(
                    set(word_tokenize(row['prompt_text'])) | set(word_tokenize(row['text']))), axis=1)

            # tqdm.pandas(desc="performing sentiment analysis")
            input_df[['sentiment_polarity', 'sentiment_subjectivity']] = input_df['text'].swifter.allow_dask_on_strings().apply(
                lambda x: pd.Series(self.sentiment_analysis(x)))

            # tqdm.pandas(desc="calculating text similarity")
            input_df['text_similarity'] = input_df.swifter.allow_dask_on_strings().apply(self.calculate_text_similarity, axis=1)

            # Calculate sentiment scores for each row
            input_df['sentiment_scores'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_sentiment_scores)

            input_df['gunning_fog'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.gunning_fog)
            input_df['flesch_kincaid_grade_level'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.flesch_kincaid_grade_level)
            input_df['count_difficult_words'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.count_difficult_words)

            # Convert sentiment_scores into individual columns
            sentiment_columns = pd.DataFrame(list(input_df['sentiment_scores']))
            input_df = pd.concat([input_df, sentiment_columns], axis=1)
            input_df['sentiment_scores_prompt'] = input_df['prompt_text'].swifter.allow_dask_on_strings().apply(self.calculate_sentiment_scores)

            # Convert sentiment_scores_prompt into individual columns
            sentiment_columns_prompt = pd.DataFrame(list(input_df['sentiment_scores_prompt']))
            sentiment_columns_prompt.columns = [col + '_prompt' for col in sentiment_columns_prompt.columns]
            input_df = pd.concat([input_df, sentiment_columns_prompt], axis=1)

            # input_df["count_unique_words"] = input_df["text"].progress_apply(
            #     self.calculate_unique_words)

            # embeddings preparation
            input_df.rename(columns={"embeddings": "stringed_embeddings"}, inplace=True)
            # tqdm.pandas(desc="embeddings transformation")
            input_df["embeddings"] = input_df["stringed_embeddings"].swifter.allow_dask_on_strings().apply(lambda x: ast.literal_eval(x))
            embeddings_length = len(input_df["embeddings"][0])
            embedding_columns = pd.DataFrame(input_df['embeddings'].to_list(),
                                             columns=[f"emb_{i}" for i in range(embeddings_length)])
            input_df = pd.concat([input_df, embedding_columns], axis=1)

            input_df = input_df.drop(columns=["summary_tokens", "prompt_tokens", "stringed_embeddings",
                                              "vader_sentiment_scores", "pos_ratios", "punctuation_ratios",
                                              "sentiment_scores", "sentiment_scores_prompt"])
            # Store to feather
            input_df.to_feather(CONFIG.storage + f"/preprocessed fold {i}.ftr")

    def run_swifter_modin(self) -> None:
        import modin.pandas as pd
        import swifter
        import ray
        ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})

        for i in range(CONFIG.num_folds):
            print(f"\nPREPROCESSING THE FOLD {i}:")
            if self.test_mode:
                input_df = pd.read_feather(path=CONFIG.storage + f"/fold {i}.ftr")[:N_ROWS]
            else:
                input_df = pd.read_feather(path=CONFIG.storage + f"/fold {i}.ftr")

            # lexical_entropy, lexical_diversity
            # tqdm.pandas(desc='lexical characteristics preparation')
            input_df['lexical_entropy'], input_df['lexical_diversity'] = zip(
                *input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_lexical_characteristics))

            # the length of a text
            input_df['text_len_symbols'] = input_df['text'].swifter.allow_dask_on_strings().apply(len)
            input_df['text_len_words'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda text: len(text.split()))
            input_df['text_len_sentences'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda text: len(sent_tokenize(text)))

            # the length of words and sentences
            input_df['sentence_length_in_words'] = input_df['text_len_words'] / input_df['text_len_sentences']
            input_df['word_length_in_symbols'] = input_df['text_len_symbols'] / input_df['text_len_words']

            # Sentiment analysis
            # tqdm.pandas(desc="sentiment analysis preparation")
            input_df['vader_sentiment_scores'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_sentiment_vader)
            input_df['vader_sentiment_positive'] = input_df['vader_sentiment_scores'].swifter.apply(lambda x: x['pos'])
            input_df['vader_sentiment_negative'] = input_df['vader_sentiment_scores'].swifter.apply(lambda x: x['neg'])
            input_df['vader_sentiment_neutral'] = input_df['vader_sentiment_scores'].swifter.apply(lambda x: x['neu'])
            input_df['vader_sentiment_compound'] = input_df['vader_sentiment_scores'].swifter.apply(lambda x: x['compound'])

            # check the level of subjectivity and polarity
            # input_df['textblob_sentiment_scores'] = input_df['text'].progress_apply(self.sentiment_analysis)
            # input_df['textblob_polarity'] = input_df['textblob_sentiment_scores'].apply(lambda x: x.polarity)
            # input_df['textblob_subjectivity'] = input_df['textblob_sentiment_scores'].apply(lambda x: x.subjectivity)

            # tqdm.pandas(desc="prompt_length preparation")
            input_df["prompt_length"] = input_df["prompt_text"].swifter.allow_dask_on_strings().apply(lambda x: len(word_tokenize(x)))
            input_df["prompt_tokens"] = input_df["prompt_text"].swifter.allow_dask_on_strings().apply(word_tokenize)

            # tqdm.pandas(desc="summary_length preparation")
            input_df["summary_length"] = input_df["text"].swifter.allow_dask_on_strings().apply(lambda x: len(word_tokenize(x)))
            input_df["summary_tokens"] = input_df["text"].swifter.allow_dask_on_strings().apply(word_tokenize)

            # Add prompt tokens into spelling checker dictionary
            # tqdm.pandas(desc="prompt_tokens preparation")
            input_df["prompt_tokens"].swifter.allow_dask_on_strings().apply(self.add_spelling_dictionary)

            # fix misspelling
            # tqdm.pandas(desc="fixed_summary_text preparation")
            input_df["fixed_summary_text"] = input_df["text"].swifter.allow_dask_on_strings().apply(self.speller)

            input_df['gunning_fog_prompt'] = input_df['prompt_text'].swifter.allow_dask_on_strings().apply(self.gunning_fog)
            input_df['flesch_kincaid_grade_level_prompt'] = input_df['prompt_text'].swifter.allow_dask_on_strings().apply(
                self.flesch_kincaid_grade_level)
            input_df['flesch_reading_ease_prompt'] = input_df['prompt_text'].swifter.allow_dask_on_strings().apply(self.flesch_reading_ease_manual)

            # count misspelling
            # tqdm.pandas(desc="splling_err_num preparation")
            input_df["splling_err_num"] = input_df["text"].swifter.allow_dask_on_strings().apply(self.spelling)

            input_df['flesch_reading_ease'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.flesch_reading_ease_manual)
            input_df['word_count'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: len(x.split()))
            input_df['sentence_length'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: len(x.split('.')))
            input_df['vocabulary_richness'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: len(set(x.split())))

            input_df['word_count2'] = [len(t.split(' ')) for t in input_df.text]
            input_df['num_unq_words'] = [len(list(set(x.lower().split(' ')))) for x in input_df.text]
            input_df['num_chars'] = [len(x) for x in input_df.text]

            # Additional features
            input_df['avg_word_length'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: np.mean([len(word) for word in x.split()]))
            input_df['comma_count'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: x.count(','))
            input_df['semicolon_count'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: x.count(';'))

            # after merge preprocess
            # tqdm.pandas(desc='length_ratio preparation')
            input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']

            # tqdm.pandas(desc='word_overlap_count preparation')
            input_df['word_overlap_count'] = input_df.swifter.apply(self.word_overlap_count, axis=1)

            # tqdm.pandas(desc='bigram_overlap_count preparation')
            input_df['bigram_overlap_count'] = input_df.swifter.apply(self.ngram_co_occurrence, args=(2,), axis=1)
            input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)

            # tqdm.pandas(desc='trigram_overlap_count preparation')
            input_df['trigram_overlap_count'] = input_df.swifter.apply(self.ngram_co_occurrence, args=(3,), axis=1)
            input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)

            # tqdm.pandas(desc='quotes_count preparation')
            input_df['quotes_count'] = input_df.swifter.apply(self.quotes_count, axis=1)

            input_df['question_count'] = input_df['text'].swifter.allow_dask_on_strings().apply(lambda x: x.count('?'))
            input_df['pos_ratios'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_pos_ratios)

            # Convert the dictionary of POS ratios into a single value (mean)
            input_df['pos_mean'] = input_df['pos_ratios'].swifter.apply(lambda x: np.mean(list(x.values())))
            input_df['punctuation_ratios'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_punctuation_ratios)

            # Convert the dictionary of punctuation ratios into a single value (sum)
            input_df['punctuation_sum'] = input_df['punctuation_ratios'].swifter.apply(lambda x: np.sum(list(x.values())))
            input_df['keyword_density'] = input_df.swifter.apply(self.calculate_keyword_density, axis=1)
            input_df['jaccard_similarity'] = input_df.swifter.allow_dask_on_strings().apply(
                lambda row: len(set(word_tokenize(row['prompt_text'])) & set(word_tokenize(row['text']))) / len(
                    set(word_tokenize(row['prompt_text'])) | set(word_tokenize(row['text']))), axis=1)

            # tqdm.pandas(desc="performing sentiment analysis")
            input_df[['sentiment_polarity', 'sentiment_subjectivity']] = input_df['text'].swifter.apply(
                lambda x: pd.Series(self.sentiment_analysis(x)))

            # tqdm.pandas(desc="calculating text similarity")
            input_df['text_similarity'] = input_df.swifter.allow_dask_on_strings().apply(self.calculate_text_similarity, axis=1)

            # Calculate sentiment scores for each row
            input_df['sentiment_scores'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.calculate_sentiment_scores)

            input_df['gunning_fog'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.gunning_fog)
            input_df['flesch_kincaid_grade_level'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.flesch_kincaid_grade_level)
            input_df['count_difficult_words'] = input_df['text'].swifter.allow_dask_on_strings().apply(self.count_difficult_words)

            # Convert sentiment_scores into individual columns
            sentiment_columns = pd.DataFrame(list(input_df['sentiment_scores']))
            input_df = pd.concat([input_df, sentiment_columns], axis=1)
            input_df['sentiment_scores_prompt'] = input_df['prompt_text'].swifter.allow_dask_on_strings().apply(self.calculate_sentiment_scores)

            # Convert sentiment_scores_prompt into individual columns
            sentiment_columns_prompt = pd.DataFrame(list(input_df['sentiment_scores_prompt']))
            sentiment_columns_prompt.columns = [col + '_prompt' for col in sentiment_columns_prompt.columns]
            input_df = pd.concat([input_df, sentiment_columns_prompt], axis=1)

            # input_df["count_unique_words"] = input_df["text"].progress_apply(
            #     self.calculate_unique_words)

            # embeddings preparation
            input_df.rename(columns={"embeddings": "stringed_embeddings"}, inplace=True)
            # tqdm.pandas(desc="embeddings transformation")
            input_df["embeddings"] = input_df["stringed_embeddings"].swifter.allow_dask_on_strings().apply(lambda x: ast.literal_eval(x))
            embeddings_length = len(input_df["embeddings"][0])
            embedding_columns = pd.DataFrame(input_df['embeddings'].to_list(),
                                             columns=[f"emb_{i}" for i in range(embeddings_length)])
            input_df = pd.concat([input_df, embedding_columns], axis=1)

            input_df = input_df.drop(columns=["summary_tokens", "prompt_tokens", "stringed_embeddings",
                                              "vader_sentiment_scores", "pos_ratios", "punctuation_ratios",
                                              "sentiment_scores", "sentiment_scores_prompt"])
            # Store to feather
            input_df.to_feather(CONFIG.storage + f"/preprocessed fold {i}.ftr")

    def run_modin(self) -> None:
        import modin.pandas as pd
        for i in range(CONFIG.num_folds):
            print(f"\nPREPROCESSING THE FOLD {i}:")
            if self.test_mode:
                input_df = pd.read_feather(path=CONFIG.storage + f"/fold {i}.ftr")[:N_ROWS]
            else:
                input_df = pd.read_feather(path=CONFIG.storage + f"/fold {i}.ftr")

            # lexical_entropy, lexical_diversity
            # tqdm.pandas(desc='lexical characteristics preparation')
            input_df['lexical_entropy'], input_df['lexical_diversity'] = zip(
                *input_df['text'].apply(self.calculate_lexical_characteristics))

            # the length of a text
            input_df['text_len_symbols'] = input_df['text'].apply(len)
            input_df['text_len_words'] = input_df['text'].apply(lambda text: len(text.split()))
            input_df['text_len_sentences'] = input_df['text'].apply(lambda text: len(sent_tokenize(text)))

            # the length of words and sentences
            input_df['sentence_length_in_words'] = input_df['text_len_words'] / input_df['text_len_sentences']
            input_df['word_length_in_symbols'] = input_df['text_len_symbols'] / input_df['text_len_words']

            # Sentiment analysis
            #tqdm.pandas(desc="sentiment analysis preparation")
            input_df['vader_sentiment_scores'] = input_df['text'].apply(self.calculate_sentiment_vader)
            input_df['vader_sentiment_positive'] = input_df['vader_sentiment_scores'].apply(lambda x: x['pos'])
            input_df['vader_sentiment_negative'] = input_df['vader_sentiment_scores'].apply(lambda x: x['neg'])
            input_df['vader_sentiment_neutral'] = input_df['vader_sentiment_scores'].apply(lambda x: x['neu'])
            input_df['vader_sentiment_compound'] = input_df['vader_sentiment_scores'].apply(lambda x: x['compound'])

            # check the level of subjectivity and polarity
            # input_df['textblob_sentiment_scores'] = input_df['text'].progress_apply(self.sentiment_analysis)
            # input_df['textblob_polarity'] = input_df['textblob_sentiment_scores'].apply(lambda x: x.polarity)
            # input_df['textblob_subjectivity'] = input_df['textblob_sentiment_scores'].apply(lambda x: x.subjectivity)

            # tqdm.pandas(desc="prompt_length preparation")
            input_df["prompt_length"] = input_df["prompt_text"].apply(lambda x: len(word_tokenize(x)))
            input_df["prompt_tokens"] = input_df["prompt_text"].apply(word_tokenize)

            # tqdm.pandas(desc="summary_length preparation")
            input_df["summary_length"] = input_df["text"].apply(lambda x: len(word_tokenize(x)))
            input_df["summary_tokens"] = input_df["text"].apply(word_tokenize)

            # Add prompt tokens into spelling checker dictionary
            # tqdm.pandas(desc="prompt_tokens preparation")
            input_df["prompt_tokens"].apply(self.add_spelling_dictionary)

            # fix misspelling
            # tqdm.pandas(desc="fixed_summary_text preparation")
            input_df["fixed_summary_text"] = input_df["text"].apply(self.speller)

            input_df['gunning_fog_prompt'] = input_df['prompt_text'].apply(self.gunning_fog)
            input_df['flesch_kincaid_grade_level_prompt'] = input_df['prompt_text'].apply(
                self.flesch_kincaid_grade_level)
            input_df['flesch_reading_ease_prompt'] = input_df['prompt_text'].apply(self.flesch_reading_ease_manual)

            # count misspelling
            # tqdm.pandas(desc="splling_err_num preparation")
            input_df["splling_err_num"] = input_df["text"].apply(self.spelling)

            input_df['flesch_reading_ease'] = input_df['text'].apply(self.flesch_reading_ease_manual)
            input_df['word_count'] = input_df['text'].apply(lambda x: len(x.split()))
            input_df['sentence_length'] = input_df['text'].apply(lambda x: len(x.split('.')))
            input_df['vocabulary_richness'] = input_df['text'].apply(lambda x: len(set(x.split())))

            input_df['word_count2'] = [len(t.split(' ')) for t in input_df.text]
            input_df['num_unq_words'] = [len(list(set(x.lower().split(' ')))) for x in input_df.text]
            input_df['num_chars'] = [len(x) for x in input_df.text]

            # Additional features
            input_df['avg_word_length'] = input_df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
            input_df['comma_count'] = input_df['text'].apply(lambda x: x.count(','))
            input_df['semicolon_count'] = input_df['text'].apply(lambda x: x.count(';'))

            # after merge preprocess
            # tqdm.pandas(desc='length_ratio preparation')
            input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']

            # tqdm.pandas(desc='word_overlap_count preparation')
            input_df['word_overlap_count'] = input_df.apply(self.word_overlap_count, axis=1)

            # tqdm.pandas(desc='bigram_overlap_count preparation')
            input_df['bigram_overlap_count'] = input_df.apply(self.ngram_co_occurrence, args=(2,), axis=1)
            input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)

            # tqdm.pandas(desc='trigram_overlap_count preparation')
            input_df['trigram_overlap_count'] = input_df.apply(self.ngram_co_occurrence, args=(3,), axis=1)
            input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)

            # tqdm.pandas(desc='quotes_count preparation')
            input_df['quotes_count'] = input_df.apply(self.quotes_count, axis=1)

            input_df['question_count'] = input_df['text'].apply(lambda x: x.count('?'))
            input_df['pos_ratios'] = input_df['text'].apply(self.calculate_pos_ratios)

            # Convert the dictionary of POS ratios into a single value (mean)
            input_df['pos_mean'] = input_df['pos_ratios'].apply(lambda x: np.mean(list(x.values())))
            input_df['punctuation_ratios'] = input_df['text'].allow_dask_on_strings().apply(self.calculate_punctuation_ratios)

            # Convert the dictionary of punctuation ratios into a single value (sum)
            input_df['punctuation_sum'] = input_df['punctuation_ratios'].apply(lambda x: np.sum(list(x.values())))
            input_df['keyword_density'] = input_df.apply(self.calculate_keyword_density, axis=1)
            input_df['jaccard_similarity'] = input_df.allow_dask_on_strings().apply(
                lambda row: len(set(word_tokenize(row['prompt_text'])) & set(word_tokenize(row['text']))) / len(
                    set(word_tokenize(row['prompt_text'])) | set(word_tokenize(row['text']))), axis=1)

            # tqdm.pandas(desc="performing sentiment analysis")
            input_df[['sentiment_polarity', 'sentiment_subjectivity']] = input_df['text'].apply(
                lambda x: pd.Series(self.sentiment_analysis(x)))

            # tqdm.pandas(desc="calculating text similarity")
            input_df['text_similarity'] = input_df.apply(self.calculate_text_similarity, axis=1)

            # Calculate sentiment scores for each row
            input_df['sentiment_scores'] = input_df['text'].apply(self.calculate_sentiment_scores)

            input_df['gunning_fog'] = input_df['text'].apply(self.gunning_fog)
            input_df['flesch_kincaid_grade_level'] = input_df['text'].apply(self.flesch_kincaid_grade_level)
            input_df['count_difficult_words'] = input_df['text'].apply(self.count_difficult_words)

            # Convert sentiment_scores into individual columns
            sentiment_columns = pd.DataFrame(list(input_df['sentiment_scores']))
            input_df = pd.concat([input_df, sentiment_columns], axis=1)
            input_df['sentiment_scores_prompt'] = input_df['prompt_text'].apply(self.calculate_sentiment_scores)

            # Convert sentiment_scores_prompt into individual columns
            sentiment_columns_prompt = pd.DataFrame(list(input_df['sentiment_scores_prompt']))
            sentiment_columns_prompt.columns = [col + '_prompt' for col in sentiment_columns_prompt.columns]
            input_df = pd.concat([input_df, sentiment_columns_prompt], axis=1)

            # input_df["count_unique_words"] = input_df["text"].progress_apply(
            #     self.calculate_unique_words)

            # embeddings preparation
            input_df.rename(columns={"embeddings": "stringed_embeddings"}, inplace=True)
            # tqdm.pandas(desc="embeddings transformation")
            input_df["embeddings"] = input_df["stringed_embeddings"].apply(lambda x: ast.literal_eval(x))
            embeddings_length = len(input_df["embeddings"][0])
            embedding_columns = pd.DataFrame(input_df['embeddings'].to_list(),
                                             columns=[f"emb_{i}" for i in range(embeddings_length)])
            input_df = pd.concat([input_df, embedding_columns], axis=1)

            input_df = input_df.drop(columns=["summary_tokens", "prompt_tokens", "stringed_embeddings",
                                              "vader_sentiment_scores", "pos_ratios", "punctuation_ratios",
                                              "sentiment_scores", "sentiment_scores_prompt"])
            # Store to feather
            input_df.to_feather(CONFIG.storage + f"/preprocessed fold {i}.ftr")

    def run_pandas(self, input_df: pd.DataFrame, queue: Queue) -> DataFrame:   # , queue: Queue
        print(multiprocessing.current_process())
        # lexical_entropy, lexical_diversity
        tqdm.pandas(desc='lexical characteristics preparation')
        input_df['lexical_entropy'], input_df['lexical_diversity'] = zip(
            *input_df['text'].progress_apply(self.calculate_lexical_characteristics))

        # the length of a text
        input_df['text_len_symbols'] = input_df['text'].apply(len)
        input_df['text_len_words'] = input_df['text'].apply(lambda text: len(text.split()))
        input_df['text_len_sentences'] = input_df['text'].apply(lambda text: len(sent_tokenize(text)))

        # the length of words and sentences
        input_df['sentence_length_in_words'] = input_df['text_len_words'] / input_df['text_len_sentences']
        input_df['word_length_in_symbols'] = input_df['text_len_symbols'] / input_df['text_len_words']

        # Sentiment analysis
        tqdm.pandas(desc="sentiment analysis preparation")
        input_df['vader_sentiment_scores'] = input_df['text'].progress_apply(self.calculate_sentiment_vader)
        input_df['vader_sentiment_positive'] = input_df['vader_sentiment_scores'].apply(lambda x: x['pos'])
        input_df['vader_sentiment_negative'] = input_df['vader_sentiment_scores'].apply(lambda x: x['neg'])
        input_df['vader_sentiment_neutral'] = input_df['vader_sentiment_scores'].apply(lambda x: x['neu'])
        input_df['vader_sentiment_compound'] = input_df['vader_sentiment_scores'].apply(lambda x: x['compound'])

        # check the level of subjectivity and polarity
        # input_df['textblob_sentiment_scores'] = input_df['text'].progress_apply(self.sentiment_analysis)
        # input_df['textblob_polarity'] = input_df['textblob_sentiment_scores'].apply(lambda x: x.polarity)
        # input_df['textblob_subjectivity'] = input_df['textblob_sentiment_scores'].apply(lambda x: x.subjectivity)

        # tqdm.pandas(desc="prompt_length preparation")
        input_df["prompt_length"] = input_df["prompt_text"].apply(lambda x: len(word_tokenize(x)))
        input_df["prompt_tokens"] = input_df["prompt_text"].apply(word_tokenize)

        # tqdm.pandas(desc="summary_length preparation")
        input_df["summary_length"] = input_df["text"].apply(lambda x: len(word_tokenize(x)))
        input_df["summary_tokens"] = input_df["text"].apply(word_tokenize)

        # Add prompt tokens into spelling checker dictionary
        tqdm.pandas(desc="prompt_tokens preparation")
        input_df["prompt_tokens"].progress_apply(self.add_spelling_dictionary)

        # fix misspelling
        tqdm.pandas(desc="fixed_summary_text preparation")
        input_df["fixed_summary_text"] = input_df["text"].progress_apply(self.speller)

        input_df['gunning_fog_prompt'] = input_df['prompt_text'].apply(self.gunning_fog)
        input_df['flesch_kincaid_grade_level_prompt'] = input_df['prompt_text'].apply(
            self.flesch_kincaid_grade_level)
        input_df['flesch_reading_ease_prompt'] = input_df['prompt_text'].apply(self.flesch_reading_ease_manual)

        # count misspelling
        tqdm.pandas(desc="splling_err_num preparation")
        input_df["splling_err_num"] = input_df["text"].progress_apply(self.spelling)

        input_df['flesch_reading_ease'] = input_df['text'].apply(self.flesch_reading_ease_manual)
        input_df['word_count'] = input_df['text'].apply(lambda x: len(x.split()))
        input_df['sentence_length'] = input_df['text'].apply(lambda x: len(x.split('.')))
        input_df['vocabulary_richness'] = input_df['text'].apply(lambda x: len(set(x.split())))

        input_df['word_count2'] = [len(t.split(' ')) for t in input_df.text]
        input_df['num_unq_words'] = [len(list(set(x.lower().split(' ')))) for x in input_df.text]
        input_df['num_chars'] = [len(x) for x in input_df.text]

        # Additional features
        input_df['avg_word_length'] = input_df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        input_df['comma_count'] = input_df['text'].apply(lambda x: x.count(','))
        input_df['semicolon_count'] = input_df['text'].apply(lambda x: x.count(';'))

        # after merge preprocess
        # tqdm.pandas(desc='length_ratio preparation')
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']

        tqdm.pandas(desc='word_overlap_count preparation')
        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)

        tqdm.pandas(desc='bigram_overlap_count preparation')
        input_df['bigram_overlap_count'] = input_df.progress_apply(self.ngram_co_occurrence, args=(2,), axis=1)
        input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)

        tqdm.pandas(desc='trigram_overlap_count preparation')
        input_df['trigram_overlap_count'] = input_df.progress_apply(self.ngram_co_occurrence, args=(3,), axis=1)
        input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)

        tqdm.pandas(desc='quotes_count preparation')
        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)

        input_df['question_count'] = input_df['text'].apply(lambda x: x.count('?'))
        input_df['pos_ratios'] = input_df['text'].apply(self.calculate_pos_ratios)

        # Convert the dictionary of POS ratios into a single value (mean)
        input_df['pos_mean'] = input_df['pos_ratios'].apply(lambda x: np.mean(list(x.values())))
        input_df['punctuation_ratios'] = input_df['text'].apply(self.calculate_punctuation_ratios)

        # Convert the dictionary of punctuation ratios into a single value (sum)
        input_df['punctuation_sum'] = input_df['punctuation_ratios'].apply(lambda x: np.sum(list(x.values())))
        input_df['keyword_density'] = input_df.apply(self.calculate_keyword_density, axis=1)
        input_df['jaccard_similarity'] = input_df.apply(
            lambda row: len(set(word_tokenize(row['prompt_text'])) & set(word_tokenize(row['text']))) / len(
                set(word_tokenize(row['prompt_text'])) | set(word_tokenize(row['text']))), axis=1)

        tqdm.pandas(desc="performing sentiment analysis")
        input_df[['sentiment_polarity', 'sentiment_subjectivity']] = input_df['text'].progress_apply(
            lambda x: pd.Series(self.sentiment_analysis(x)))

        tqdm.pandas(desc="calculating text similarity")
        input_df['text_similarity'] = input_df.progress_apply(self.calculate_text_similarity, axis=1)

        # Calculate sentiment scores for each row
        input_df['sentiment_scores'] = input_df['text'].apply(self.calculate_sentiment_scores)

        input_df['gunning_fog'] = input_df['text'].apply(self.gunning_fog)
        input_df['flesch_kincaid_grade_level'] = input_df['text'].apply(self.flesch_kincaid_grade_level)
        input_df['count_difficult_words'] = input_df['text'].apply(self.count_difficult_words)

        # Convert sentiment_scores into individual columns
        sentiment_columns = pd.DataFrame(list(input_df['sentiment_scores']))
        input_df = pd.concat([input_df, sentiment_columns], axis=1)
        input_df['sentiment_scores_prompt'] = input_df['prompt_text'].apply(self.calculate_sentiment_scores)

        # Convert sentiment_scores_prompt into individual columns
        sentiment_columns_prompt = pd.DataFrame(list(input_df['sentiment_scores_prompt']))
        sentiment_columns_prompt.columns = [col + '_prompt' for col in sentiment_columns_prompt.columns]
        input_df = pd.concat([input_df, sentiment_columns_prompt], axis=1)

        # input_df["count_unique_words"] = input_df["text"].progress_apply(
        #     self.calculate_unique_words)

        # embeddings preparation
        input_df.rename(columns={"embeddings": "stringed_embeddings"}, inplace=True)
        tqdm.pandas(desc="embeddings transformation")
        input_df["embeddings"] = input_df["stringed_embeddings"].progress_apply(lambda x: ast.literal_eval(x))
        embeddings_length = len(input_df["embeddings"][0])
        embedding_columns = pd.DataFrame(input_df['embeddings'].to_list(),
                                         columns=[f"emb_{i}" for i in range(embeddings_length)])
        input_df = pd.concat([input_df, embedding_columns], axis=1)

        input_df = input_df.drop(columns=["summary_tokens", "prompt_tokens", "stringed_embeddings",
                                          "vader_sentiment_scores", "pos_ratios", "punctuation_ratios",
                                          "sentiment_scores", "sentiment_scores_prompt"])
        queue.put(input_df)
        # return input_df

    def run_2(self, prompts: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
        # before merge preprocess
        prompts["prompt_length"] = prompts["prompt_text"].apply(
            lambda x: len(word_tokenize(x))
        )
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: word_tokenize(x)
        )

        summaries["summary_length"] = summaries["text"].apply(
            lambda x: len(word_tokenize(x))
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: word_tokenize(x)
        )

        # Add prompt tokens into spelling checker dictionary
        prompts["prompt_tokens"].apply(
            lambda x: self.add_spelling_dictionary(x)
        )

        prompts['gunning_fog_prompt'] = prompts['prompt_text'].apply(self.gunning_fog)
        prompts['flesch_kincaid_grade_level_prompt'] = prompts['prompt_text'].apply(
            self.flesch_kincaid_grade_level)
        prompts['flesch_reading_ease_prompt'] = prompts['prompt_text'].apply(self.flesch_reading_ease_manual)

        # count misspelling
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)

        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")
        input_df['flesch_reading_ease'] = input_df['text'].apply(self.flesch_reading_ease_manual)
        input_df['word_count'] = input_df['text'].apply(lambda x: len(x.split()))
        input_df['sentence_length'] = input_df['text'].apply(lambda x: len(x.split('.')))
        input_df['vocabulary_richness'] = input_df['text'].apply(lambda x: len(set(x.split())))

        input_df['word_count2'] = [len(t.split(' ')) for t in input_df.text]
        input_df['num_unq_words'] = [len(list(set(x.lower().split(' ')))) for x in input_df.text]
        input_df['num_chars'] = [len(x) for x in input_df.text]

        # Additional features
        input_df['avg_word_length'] = input_df['text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]))
        input_df['comma_count'] = input_df['text'].apply(lambda x: x.count(','))
        input_df['semicolon_count'] = input_df['text'].apply(lambda x: x.count(';'))

        # after merge preprocess
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']

        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)
        input_df['bigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(2,), axis=1
        )
        input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)

        input_df['trigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(3,), axis=1
        )
        input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)

        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)

        input_df['exclamation_count'] = input_df['text'].apply(lambda x: x.count('!'))
        input_df['question_count'] = input_df['text'].apply(lambda x: x.count('?'))
        input_df['pos_ratios'] = input_df['text'].apply(self.calculate_pos_ratios)

        # Convert the dictionary of POS ratios into a single value (mean)
        input_df['pos_mean'] = input_df['pos_ratios'].apply(lambda x: np.mean(list(x.values())))
        input_df['punctuation_ratios'] = input_df['text'].apply(self.calculate_punctuation_ratios)

        # Convert the dictionary of punctuation ratios into a single value (sum)
        input_df['punctuation_sum'] = input_df['punctuation_ratios'].apply(lambda x: np.sum(list(x.values())))
        input_df['keyword_density'] = input_df.apply(self.calculate_keyword_density, axis=1)
        input_df['jaccard_similarity'] = input_df.apply(
            lambda row: len(set(word_tokenize(row['prompt_text'])) & set(word_tokenize(row['text']))) / len(
                set(word_tokenize(row['prompt_text'])) | set(word_tokenize(row['text']))), axis=1)
        tqdm.pandas(desc="Performing Sentiment Analysis")
        input_df[['sentiment_polarity', 'sentiment_subjectivity']] = input_df['text'].progress_apply(
            lambda x: pd.Series(self.sentiment_analysis(x))
        )
        tqdm.pandas(desc="Calculating Text Similarity")
        input_df['text_similarity'] = input_df.progress_apply(self.calculate_text_similarity, axis=1)
        # Calculate sentiment scores for each row
        input_df['sentiment_scores'] = input_df['text'].apply(self.calculate_sentiment_scores)

        input_df['gunning_fog'] = input_df['text'].apply(self.gunning_fog)
        input_df['flesch_kincaid_grade_level'] = input_df['text'].apply(self.flesch_kincaid_grade_level)
        input_df['count_difficult_words'] = input_df['text'].apply(self.count_difficult_words)

        # Convert sentiment_scores into individual columns
        sentiment_columns = pd.DataFrame(list(input_df['sentiment_scores']))
        input_df = pd.concat([input_df, sentiment_columns], axis=1)
        input_df['sentiment_scores_prompt'] = input_df['prompt_text'].apply(self.calculate_sentiment_scores)
        # Convert sentiment_scores_prompt into individual columns
        sentiment_columns_prompt = pd.DataFrame(list(input_df['sentiment_scores_prompt']))
        sentiment_columns_prompt.columns = [col + '_prompt' for col in sentiment_columns_prompt.columns]
        input_df = pd.concat([input_df, sentiment_columns_prompt], axis=1)
        columns = ['pos_ratios', 'sentiment_scores', 'punctuation_ratios', 'sentiment_scores_prompt']
        cols_to_drop = [col for col in columns if col in input_df.columns]
        if cols_to_drop:
            input_df = input_df.drop(columns=cols_to_drop)

        print(cols_to_drop)
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])


def group_folds_in_a_single_df() -> pd.DataFrame:
    """
    Downloads data from 'preprocessed fold xxx.ftr' with a feature engineered data +
    separately placed by columns embeddings ,
    """
    total_df = pd.DataFrame()
    for i in range(CONFIG.num_folds):
        fold_df = pd.read_feather(CONFIG.storage + f"/preprocessed fold {i}.ftr")
        fold_df["fold"] = i
        if total_df.empty:
            total_df = fold_df
        else:
            total_df = pd.concat([total_df, fold_df], axis=0, ignore_index=True)
    return total_df


def split_data_on_train_test(data: pd.DataFrame, fold_nummer: int, target_name: str) -> (
        tuple)[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train = data[data["fold"] != fold_nummer].drop(columns=drop_columns + CONFIG.data.targets)
    y_train = data[data["fold"] != fold_nummer][target_name]

    x_test = data[data["fold"] == fold_nummer].drop(columns=drop_columns + CONFIG.data.targets)
    y_test = data[data["fold"] == fold_nummer][target_name]
    return x_train, x_test, y_train, y_test


def normalize_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_df[features_for_norm] = scaler.fit_transform(train_df[features_for_norm])
    test_df[features_for_norm] = scaler.transform(test_df[features_for_norm])
    return train_df, test_df, scaler


def clean_lexile(lexile):
    """
    Function to clean lexile feature
    Args:
        lexile (str or float): The lexile measure as a string or a float

    Returns:
        int or np.nan: The cleaned lexile measure as an integer, or np.nan for 'Non-Prose' or 'nan' values
    """
    if pd.isnull(lexile):
        return np.nan
    elif isinstance(lexile, str):
        if lexile == 'Non-Prose':
            return np.nan
        else:
            # Remove the 'L' at the end and convert to integer
            return int(lexile.rstrip('L'))
    else:
        # If lexile is a float (or any non-string data type), convert to int and return
        return int(lexile)


def classify_author(spacy_ner_model, author):
    # Process the text
    doc = spacy_ner_model(author)

    # Check if any of the entities are labeled as 'PERSON'
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return 'person'

    # If no 'PERSON' entity is found, return 'org'
    return 'org'


def encode_author_type(df):
    """
    Function to encode author_type feature
    Args:
        df (pd.DataFrame): The DataFrame with 'author_type' column

    Returns:
        pd.DataFrame: The DataFrame with 'author_type' replaced with numerical values
    """
    le = LabelEncoder()
    df['author_type'] = le.fit_transform(df['author_type'])
    return df


def clean_grade(df):
    """
    Function to clean grade feature
    Args:
        df (pd.DataFrame): The DataFrame with 'grade' column

    Returns:
        pd.DataFrame: The DataFrame with 'grade' replaced with integer values
    """
    df['grade'] = df['grade'].astype(str).str.replace('rd Grade', '')
    df['grade'] = df['grade'].str.replace('th Grade', '')
    df['grade'] = df['grade'].apply(lambda x: int(x) if x.isdigit() else 0)
    return df


def group_and_encode_genre(df):
    """
    Function to group and encode genre feature
    Args:
        df (pd.DataFrame): The DataFrame with 'genre' column

    Returns:
        pd.DataFrame: The DataFrame with 'genre' replaced with grouped and encoded values
    """
    genre_map = {
        'Fiction': ['Poem', 'Short Story', 'Folktale', 'Fantasy', 'Science Fiction', 'Allegory',
                    'Fiction - General', 'Fable', 'Myth', 'Historical Fiction', 'Magical Realism', 'Drama'],
        'Non-Fiction': ['Informational Text', 'Non-Fiction - General', 'Biography', 'Essay', 'Memoir', 'Interview',
                        'Psychology', 'Primary Source Document', 'Autobiography'],
        'News & Opinion': ['News', 'Opinion'],
        'Historic & Legal': ['Historical Document', 'Legal Document', 'Letter'],
        'Philosophy & Religion': ['Speech', 'Religious Text', 'Satire', 'Political Theory', 'Philosophy']
    }

    # Reverse the genre_map dictionary for mapping
    reverse_genre_map = {genre: key for key, values in genre_map.items() for genre in values}

    df['genre_big_group'] = df['genre'].map(reverse_genre_map)

    # If the genre is not found in the map, assign it to 'Other'
    df['genre_big_group'] = df['genre_big_group'].fillna('Other')

    le = LabelEncoder()
    df['genre_big_group_encode'] = le.fit_transform(df['genre_big_group'])

    return df


def preprocess_and_join(df1, df2, df1_title_col, df2_title_col, grade_col):
    # Copy dataframes to avoid modifying the originals
    df1 = df1.copy()
    df2 = df2.copy()

    # Preprocess titles
    df1[df1_title_col] = df1[df1_title_col].str.replace('"', '').str.strip()
    df2[df2_title_col] = df2[df2_title_col].str.replace('"', '').str.strip()

    # Remove duplicate grades
    df2 = df2.drop_duplicates(subset=df2_title_col, keep='first')

    # Join dataframes
    merged_df = df1.merge(df2, how='left', left_on=df1_title_col, right_on=df2_title_col)

    # Postprocess grades
    merged_df[grade_col] = merged_df[grade_col].fillna(0)
    merged_df[grade_col] = merged_df[grade_col].astype(int).astype('category')

    return merged_df

