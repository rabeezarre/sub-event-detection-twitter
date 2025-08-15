import os
import re
import time
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import os
import re
import time
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from joblib import Parallel, delayed
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2


# Not used
def focal_loss(gamma=2., alpha=0.25):

    def focal_loss_fixed(y_true, y_pred):

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * \
            K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        focal_loss = weight * K.pow(1 - y_pred, gamma) * cross_entropy

        return K.mean(focal_loss)

    return focal_loss_fixed


class ModelAnalyzer:
    def __init__(self, model, preprocessor, vectorizer):
        self.model = model
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.analysis_results = {}

    def analyze_predictions(self, X, y_true, y_pred, feature_names=None):
        # Get predictions
        errors = np.where(y_true != y_pred)[0]
        raw_predictions = self.model.predict(X).flatten()
        confidence_stats = self._analyze_prediction_confidence(
            raw_predictions, y_true)
        feature_importance = self._analyze_feature_importance(
            X, y_true, feature_names)

        self.analysis_results = {
            'error_rate': 1 - accuracy_score(y_true, y_pred),
            'error_indices': errors,
            'feature_importance': feature_importance,
            'confidence_stats': confidence_stats,
            'class_distribution': np.bincount(y_true.astype(int))
        }

        self._save_analysis_to_files()
        return self.analysis_results

    def _analyze_feature_importance(self, X, y_true, feature_names):
        base_score = self.model.evaluate(X, y_true, verbose=0)[1]
        importance_scores = []

        for i in range(X.shape[1]):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            new_score = self.model.evaluate(X_permuted, y_true, verbose=0)[1]
            importance = base_score - new_score
            importance_scores.append(importance)

        return list(zip(feature_names if feature_names else range(len(importance_scores)),
                        importance_scores))

    def _analyze_prediction_confidence(self, raw_predictions, y_true):
        mask_correct = (y_true == (raw_predictions > 0.5))
        confidence_correct = raw_predictions[mask_correct]
        confidence_incorrect = raw_predictions[~mask_correct]

        return {
            'avg_confidence_correct': float(np.mean(np.abs(confidence_correct - 0.5) * 2)),
            'avg_confidence_incorrect': float(np.mean(np.abs(confidence_incorrect - 0.5) * 2)),
            'high_confidence_errors': int(np.sum(np.abs(confidence_incorrect - 0.5) > 0.4))
        }

    def _save_analysis_to_files(self):
        os.makedirs('analysis_output', exist_ok=True)

        with open('analysis_output/error_analysis.txt', 'w') as f:
            f.write(f"Error Analysis\n")
            f.write("=" * 50 + "\n")
            f.write(f"Error rate: {self.analysis_results['error_rate']:.4f}\n")
            f.write(
                f"Number of errors: {len(self.analysis_results['error_indices'])}\n")
            f.write(
                f"Class distribution: {self.analysis_results['class_distribution']}\n")

        # Save confidence analysis
        with open('analysis_output/confidence_analysis.txt', 'w') as f:
            f.write("Confidence Analysis\n")
            f.write("=" * 50 + "\n")
            for metric, value in self.analysis_results['confidence_stats'].items():
                f.write(f"{metric}: {value:.4f}\n")

        # Save feature importance
        with open('analysis_output/feature_importance.txt', 'w') as f:
            f.write("Feature Importance Analysis\n")
            f.write("=" * 50 + "\n")
            sorted_features = sorted(self.analysis_results['feature_importance'],
                                     key=lambda x: abs(x[1]), reverse=True)
            for feature, importance in sorted_features:
                f.write(f"{feature}: {importance:.4f}\n")


class TweetPreprocessor:
    def __init__(self):

        # Download necessary data
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('punkt_tab')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Initialize the dictionaries and patterns
        self._init_soccer_abbrev()
        self._init_team_names()
        self._init_common_abbrev()
        self._init_emoji_mappings()
        self._init_patterns()
        self._init_promo_patterns()

    def _init_soccer_abbrev(self):
        """Initialize soccer abbreviations"""
        self.soccer_abbrev = {
            'ht': 'halftime', 'ft': 'fulltime', 'et': 'extra time',
            'pk': 'penalty kick', 'pens': 'penalties', 'aet': 'after extra time',
            'ko': 'kickoff',

            'pen': 'penalty', 'yc': 'yellow card', 'rc': 'red card',
            'fk': 'free kick', 'ck': 'corner kick', 'ps': 'penalty save',
            'var': 'video assistant referee', 'gs': 'goal scorer',

            'gk': 'goalkeeper', 'def': 'defender', 'mid': 'midfielder',
            'fwd': 'forward', 'cb': 'center back', 'lb': 'left back',
            'rb': 'right back', 'cdm': 'defensive midfielder',
            'cam': 'attacking midfielder', 'lw': 'left wing',
            'rw': 'right wing', 'st': 'striker', 'ref': 'referee',

            'motm': 'man of the match',
            'cs': 'clean sheet', 'poss': 'possession',
            'sot': 'shots on target', 'mins': 'minutes',
            'agg': 'aggregate', 'pts': 'points',
            'assists': 'assists', 'apps': 'appearances',
            'susp': 'suspended',

            'ucl': 'champions league', 'uel': 'europa league',
            'wc': 'world cup', 'qual': 'qualifiers',
            'intl': 'international', 'epl': 'premier league',
        }

    def _init_team_names(self):
        """Initialize team nicknames"""
        self.team_names = {
            # South American teams
            'selecao': 'brazil', 'canarinho': 'brazil',
            'verde amarela': 'brazil', 'albiceleste': 'argentina',
            'la albiceleste': 'argentina', 'la roja': 'chile',
            'los incas': 'peru', 'la celeste': 'uruguay',
            'la vinotinto': 'venezuela', 'los cafeteros': 'colombia',

            # European teams
            'les bleus': 'france', 'azzurri': 'italy',
            'mannschaft': 'germany', 'die mannschaft': 'germany',
            'nationalelf': 'germany', 'oranje': 'netherlands',
            'clockwork orange': 'netherlands', 'three lions': 'england',
            'la roja': 'spain', 'la furia roja': 'spain',
            'selecao das quinas': 'portugal', 'red devils': 'belgium',
            'vatreni': 'croatia',

            # African teams
            'super eagles': 'nigeria', 'black stars': 'ghana',
            'atlas lions': 'morocco', 'teranga lions': 'senegal',
            'pharaohs': 'egypt', 'elephants': 'ivory coast',
            'indomitable lions': 'cameroon', 'carthage eagles': 'tunisia',

            # Major club teams
            'red devils': 'manchester united', 'gunners': 'arsenal',
            'blues': 'chelsea', 'reds': 'liverpool',
            'spurs': 'tottenham', 'citizens': 'manchester city',
            'toffees': 'everton', 'blaugrana': 'barcelona',
            'merengues': 'real madrid', 'bavarians': 'bayern munich'
        }

    def _init_common_abbrev(self):
        """Initialize common social media abbreviations"""
        self.common_abbrev = {
            'u': 'you', 'r': 'are', 'ur': 'your', 'n': 'and',
            'bc': 'because', 'b4': 'before', 'br': 'brother',
            'cuz': 'because', 'dm': 'direct message',
            'tbh': 'to be honest', 'imo': 'in my opinion',
            'idk': 'i do not know', 'irl': 'in real life',
            'omg': 'oh my god', 'omw': 'on my way',
            'rn': 'right now', 'tbf': 'to be fair',

            'lol': 'laughing out loud', 'lmao': 'laughing',
            'rofl': 'rolling floor laughing', 'ffs': 'for sake',
            'smh': 'shaking my head', 'ngl': 'not going to lie',
            'gg': 'good game', 'wp': 'well played',

            'af': 'very', 'rly': 'really', 'sry': 'sorry',
            'pls': 'please', 'plz': 'please', 'thx': 'thanks',
            'ty': 'thank you', 'wb': 'welcome back',

            'goat': 'greatest of all time', 'mvp': 'most valuable player',
            'fwiw': 'for what it is worth', 'ftw': 'for the win',
            'iirc': 'if i remember correctly'
        }

    def _init_emoji_mappings(self):
        """Initialize emoji to text mappings"""
        self.emoji_mappings = {
            # Soccer specific mapping
            '⚽': ' goal ', '🎯': ' goal ', '🥅': ' goal ',
            '🔴': ' red card ', '🟡': ' yellow card ',
            '🧤': ' save ', '👐': ' goalkeeper ',
            '🔄': ' substitution ', '🏃': ' running ', '💫': ' skill ',

            # Reactions mapping
            '🔥': ' amazing ', '👏': ' applause ', '🏆': ' champion ',
            '💪': ' strong ', '🌟': ' star ', '👑': ' victory ',
            '😭': ' sad ', '🎉': ' celebration ', '🙌': ' celebration ',
            '🤩': ' spectacular ', '😱': ' shocking ',

            # Match status mapping
            '✅': ' win ', '❌': ' loss ', '🤝': ' draw ',
            '⏰': ' time ', '📋': ' lineup ', '🔍': ' analysis ',
            '📊': ' stats ', '📈': ' improving ', '📉': ' declining '
        }

    def _init_patterns(self):
        """Initialize regex patterns"""
        self.patterns = {
            'scores': re.compile(r'\d{1,2}[-:]\d{1,2}'),
            'hashtags': re.compile(r'#(\w+)'),
            'mentions': re.compile(r'@(\w+)'),
            'urls': re.compile(r'http\S+|www.\S+'),
            # starts by stream or watch or live or live...stream
            'streaming_links': re.compile(r'stream\S*|watch\S*live|live\S*stream'),
            'score_updates': re.compile(r'score update|live score|match update'),
            'elongated': re.compile(r'(\w)\1{2,}'),
            'numbers': re.compile(r'\d+'),
            'special_chars': re.compile(r'[^\w\s]'),
        }
        # detect words with a letter repeated consecutively 3 times
        self.elongation_pattern = re.compile(r'(.)\1{2,}', re.DOTALL)

    def _init_promo_patterns(self):
        """Initialize promotional content patterns"""
        self.promo_patterns = {
            'giveaway': re.compile(r'give\s*away|giving\s*away|win\s+a|winner|contest|free'),
            'entry_rules': re.compile(r'follow\s+(?:and|&)\s+rt|retweet\s+to|rt\s+to|to\s+enter'),

            'betting': re.compile(r'bet|odds|betting|gambling|bookmaker'),
            'betting_sites': re.compile(r'betfair|paddy\s*power'),

            'marketing': re.compile(r'discount|promo|offer|sale|buy|purchase|order'),
            'calls_to_action': re.compile(r'click\s+here|sign\s+up|register|subscribe'),

            'social_promotion': re.compile(r'follow\s+us|subscribe|like|share'),
            'social_growth': re.compile(r'gain\s+followers|grow\s+your|increase\s+your')
        }

    def is_promotional(self, text):
        """Check if text is promotional content"""

        for pattern_name, pattern in self.promo_patterns.items():
            if pattern.search(text):
                return True

        # Filter out if too many hashtags, mentions or links
        if text.count('#') > 3:
            return True
        if text.count('@') > 3:
            return True
        if text.count('http') > 1:
            return True

        return False

    def normalize_elongated(self, text):
        """Replace elongated words by their original writing"""
        return self.elongation_pattern.sub(r'\1', text)

    def preprocess(self, text):
        """Preprocess the text"""
        text = text.lower()

        # remove retweets
        remove_retweet_mention = re.compile(r'\brt @[\w._]+:?(\s+|$)')
        if remove_retweet_mention.match(text):
            return ""

        # remove promotional content
        if self.is_promotional(text):
            return ""

        # Replace scores with placeholders
        scores = self.patterns['scores'].finditer(text)
        score_placeholders = {}
        for i, match in enumerate(scores):
            placeholder = f" SCORE_{i} "
            score_placeholders[placeholder] = match.group()
            text = text.replace(match.group(), placeholder)

        # Remove unwanted content
        text = self.patterns['hashtags'].sub(r'\1', text)
        text = self.patterns['urls'].sub('', text)
        text = self.patterns['streaming_links'].sub('', text)
        text = self.patterns['score_updates'].sub('', text)
        text = self.patterns['numbers'].sub('', text)
        text = self.patterns['special_chars'].sub(' ', text)

        # Replace emojis
        for emoji, replacement in self.emoji_mappings.items():
            text = text.replace(emoji, f" {replacement} ")

        # Normalize elongated words
        text = self.normalize_elongated(text)

        # Remove special characters
        text = self.patterns['special_chars'].sub(' ', text)

        # Tokenize and process words
        words = word_tokenize(text)
        processed_words = []

        for word in words:
            # Try soccer abbreviations first, then team names, then common abbreviations
            word = self.soccer_abbrev.get(word,
                                          self.team_names.get(word,
                                                              self.common_abbrev.get(word, word)))

            if word not in self.stop_words:
                word = self.lemmatizer.lemmatize(word)
                processed_words.append(word)

        text = ' '.join(processed_words)

        # Restore saved scores
        for placeholder, score in score_placeholders.items():
            text = text.replace(placeholder, f" {score} ")

        return text

    def process_batch(self, texts, batch_size=1000, return_stats=False):
        """Process a batch of tweets (with optional statistics)"""
        processed_texts = []
        filtered_count = 0
        filter_reasons = {}

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                is_promo, reason = self.is_promotional(text)
                if is_promo:
                    filtered_count += 1
                    filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                    processed_texts.append("")
                else:
                    processed_texts.append(self.preprocess(text))

        if return_stats:
            stats = {
                'total_tweets': len(texts),
                'filtered_tweets': filtered_count,
                'filter_percentage': (filtered_count / len(texts)) * 100,
                'filter_reasons': filter_reasons
            }
            return processed_texts, stats

        return processed_texts


class TweetVectorizer:
    """ Vectorize tweets using pre-trained GloVe embeddings, with domain-specific weighting for soccer related terms"""

    def __init__(self, vector_size=200):
        self.embeddings_model = api.load("glove-twitter-200")
        self.vector_size = vector_size

        self.domain_weights = {
            'goal': 2.0, 'score': 1.8, 'kick': 1.5, 'penalty': 2.0,
            'card': 1.5, 'yellow': 1.5, 'red': 1.5, 'half': 1.5,
            'time': 1.3, 'whistle': 1.5, 'referee': 1.3, 'foul': 1.5,
            'match': 1.2, 'game': 1.2, 'ball': 1.2, 'shot': 1.5,
            'save': 1.5, 'corner': 1.3, 'offside': 1.5, 'injury': 1.5
        }

    def vectorize(self, tweet):
        """Converts a  tweet into a weighted vector representation"""

        words = tweet.split()
        vectors = []
        weights = []

        for word in words:
            if word in self.embeddings_model:
                vectors.append(self.embeddings_model[word])
                weight = next((w for term, w in self.domain_weights.items()
                               if term in word), 1.0)
                weights.append(weight)

        if not vectors:
            return np.zeros(self.vector_size)

        vectors = np.array(vectors)
        weights = np.array(weights).reshape(-1, 1)
        return np.sum(vectors * weights, axis=0) / np.sum(weights)

    def vectorize_batch(self, tweets, batch_size=1000):
        """Vectorizes a batch of tweets"""
        vectors = []
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i + batch_size]
            batch_vectors = [self.vectorize(tweet) for tweet in batch]
            vectors.extend(batch_vectors)
        return np.array(vectors)


class TemporalFeatures:
    """Extract temporal features from tweet data, based on a sliding window of a specified size"""

    def __init__(self, window_size=60):  # 60 second window
        self.window_size = window_size

    def extract_temporal_features(self, df):
        """Extract temporal features from tweet data"""

        # Sort by timestamp
        df = df.sort_values('Timestamp')

        features = {}

        # Group by match and period
        for (match_id, period_id), group in df.groupby(['MatchID', 'PeriodID']):
            timestamps = group['Timestamp'].values

            for idx, tweet_time in enumerate(timestamps):

                # Define the window boundaries
                window_start = tweet_time - self.window_size
                window_end = tweet_time

                # Count tweets in window
                tweets_in_window = len(timestamps[(timestamps >= window_start) &
                                                  (timestamps < window_end)])

                # Calculate rate of change in tweet frequency
                prev_window_start = window_start - self.window_size
                prev_window_tweets = len(timestamps[(timestamps >= prev_window_start) &
                                                    (timestamps < window_start)])

                frequency_change = tweets_in_window - prev_window_tweets

                # Store features
                tweet_id = group.iloc[idx]['ID']
                features[tweet_id] = {
                    'tweets_in_window': tweets_in_window,
                    'frequency_change': frequency_change
                }

        # Convert to DataFrame
        temporal_features = pd.DataFrame.from_dict(features, orient='index')
        temporal_features.index.name = 'ID'
        temporal_features = temporal_features.reset_index()

        return temporal_features


def process_data(file_paths, preprocessor, vectorizer, temporal_processor, batch_size=1000):
    """Processes a list of csv files containing tweet data"""
    all_features = []

    li = []
    for filename in file_paths:
        df = pd.read_csv(filename)
        temporal_features = temporal_processor.extract_temporal_features(df)

        # Preprocess tweets
        processed_tweets = []
        for i in range(0, len(df), batch_size):
            batch = df['Tweet'].iloc[i:i + batch_size]
            processed = [preprocessor.preprocess(tweet) for tweet in batch]
            processed_tweets.extend(processed)

        df['Tweet'] = processed_tweets

        tweet_vectors = vectorizer.vectorize_batch(
            df['Tweet'].tolist(), batch_size)

        features = pd.DataFrame(tweet_vectors)
        features['MatchID'] = df['MatchID']
        features['PeriodID'] = df['PeriodID']
        features['ID'] = df['ID']

        features = features.merge(temporal_features, on='ID')

        if 'EventType' in df.columns:
            features['EventType'] = df['EventType']

        all_features.append(features)

    if not all_features:
        raise ValueError("No valid data was processed")

    return pd.concat(all_features, ignore_index=True)


def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[
                  'accuracy', 'AUC', 'Precision', 'Recall'])

    return model


def main():
    preprocessor = TweetPreprocessor()
    vectorizer = TweetVectorizer()
    temporal_processor = TemporalFeatures(window_size=60)

    train_tweets_dir = os.path.join(os.path.dirname(__file__), "train_tweets")
    train_files = [os.path.join(train_tweets_dir, f)
                   for f in os.listdir(train_tweets_dir)]

    try:
        train_features = process_data(
            train_files, preprocessor, vectorizer, temporal_processor)
        train_features = train_features.groupby(
            ['MatchID', 'PeriodID', 'ID']).mean().reset_index()

        feature_names = [f'feature_{i}' for i in range(
            train_features.shape[1] - 4)]

        X = train_features.drop(
            columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
        y = train_features['EventType'].values

        train_indices, test_indices = train_test_split(
            range(len(X)), test_size=0.2, random_state=42)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)

        model = create_model(X.shape[1])

        # Tried with this, we got a worse accuracy

        # class_weights = compute_class_weight(
        #     'balanced',
        #     classes=np.unique(y_train),
        #     y=y_train
        # )

        # class_weights_dict = dict(zip(np.unique(y_train), class_weights))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True)

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            # class_weight=class_weights_dict,
            epochs=100,
            batch_size=64,
            callbacks=[early_stopping]
        )

        y_pred = (model.predict(X_test) > 0.5).astype(float)
        analyzer = ModelAnalyzer(model, preprocessor, vectorizer)

        analysis_results = analyzer.analyze_predictions(
            X_test, y_test, y_pred, feature_names)
        print(f'Analysis files generated in analysis_output directory')

        print(f'Test Accuracy: {accuracy_score(y_test, y_pred)}')
        print("F1 Score:", f1_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        eval_tweets_dir = os.path.join(
            os.path.dirname(__file__), "eval_tweets")
        eval_files = [os.path.join(eval_tweets_dir, f)
                      for f in os.listdir(eval_tweets_dir)]

        eval_features = process_data(
            eval_files, preprocessor, vectorizer, temporal_processor)
        eval_features = eval_features.groupby(
            ['MatchID', 'PeriodID', 'ID']).mean().reset_index()

        X_eval = eval_features.drop(
            columns=['MatchID', 'PeriodID', 'ID']).values
        predictions = (model.predict(X_eval) > 0.5).astype(float)

        submission = pd.DataFrame({
            'ID': eval_features['ID'].values,
            'EventType': predictions.flatten()
        })
        submission.to_csv('new_predictions.csv', index=False)

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
