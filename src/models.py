import numpy as np
import torch.nn as nn
from config import NUM_LAYERS, DROPOUT, MODEL_REGISTRY
from collections import defaultdict
# from pomegranate.hmm import DenseHMM, State, DiscreteDistribution

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def generate_model(model_name: str, *args, **kwargs):
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Model '{model_name}' is not recognized.")
    return model_cls(*args, **kwargs)

# @register_model("LSTMArabicModel")
# class LSTMArabicModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, PAD):
#         super(LSTMArabicModel, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.embedding = nn.Embedding(
#             vocab_size, embedding_dim, padding_idx=PAD)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim,
#                             batch_first=True, bidirectional=True,
#                             num_layers=NUM_LAYERS, dropout=DROPOUT)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         lstm_out, _ = self.lstm(embedded)
#         output = self.fc(lstm_out)
#         return output

@register_model("HMMArabicModel")
class HMMArabicModel(nn.Module):
    def __init__(self, num_states, num_observations, pad_state_id=None):
        super(HMMArabicModel, self).__init__()
        self.num_states = num_states
        self.num_observations = num_observations
        self.pad_state_id = pad_state_id
        
        self.log_pi = None             # (num_states,) initial state log probabilities
        self.log_transition = None     # (num_states, num_states)
        self.log_emission = None       # (num_states, num_observations)

    def fit(self, seq_obs, seq_states, laplace=1.0):
        """
        seq_obs: list of observation sequences (char ids)
        seq_states: list of corresponding state sequences (diac ids)
        laplace: add-k smoothing constant
        """
        pi_counts = np.zeros(self.num_states, dtype=np.float64)
        transition_counts = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        emission_counts = np.zeros((self.num_states, self.num_observations), dtype=np.float64)

        total_sequences = 0
        for obs, states in zip(seq_obs, seq_states):
            #uniform length check
            if len(obs) == 0 or len(obs) != len(states):
                continue
            total_sequences += 1

            # initial state
            s0 = states[0]
            if self.is_pad(s0):
                continue
            pi_counts[s0] += 1.0
            
            # emissions and transitions
            for i in range(len(obs)):
                s = states[i]
                o = obs[i]
                if self.is_pad(s) or o is None:
                    continue
                emission_counts[s, o] += 1.0
                if i + 1 < len(obs):
                    s_next = states[i + 1]
                    if not self.is_pad(s_next):
                        transition_counts[s, s_next] += 1.0
        # apply Laplace smoothing then normalize
        pi_sm = pi_counts + laplace
        self.log_pi = np.log(pi_sm / pi_sm.sum())

        log_transition = transition_counts + laplace
        transition_row_sums = log_transition.sum(axis=1, keepdims=True)
        # avoid divide by zero
        transition_row_sums[transition_row_sums == 0] = 1.0
        self.log_transition = np.log(log_transition / transition_row_sums)

        log_emission = emission_counts + laplace
        emission_row_sums = log_emission.sum(axis=1, keepdims=True)
        emission_row_sums[emission_row_sums == 0] = 1.0
        self.log_emission = np.log(log_emission / emission_row_sums)

        return self
    
    def is_pad(self, state_id):
        return self.pad_state_id is not None and state_id == self.pad_state_id

    def viterbi(self, obs_seq):
        """
        obs_seq: list of observation ids (ints)
        returns: best state sequence (list of state ids)
        """
        T = len(obs_seq)
        N = self.num_states
        if T == 0:
            return []

        # Use -inf for impossible
        neginf = -1e300

        # delta[t, i] = max log prob of a path ending in state i at time t
        delta = np.full((T, N), neginf, dtype=np.float64)
        psi = np.zeros((T, N), dtype=np.int32)

        # init
        o0 = obs_seq[0]
        # if observation index out of range, treat emission log-prob as neginf
        emit0 = self.log_emission[:, o0] if 0 <= o0 < self.num_observations else np.full(N, neginf)
        delta[0, :] = self.log_pi + emit0
        psi[0, :] = 0

        for t in range(1, T):
            ot = obs_seq[t]
            emit_t = self.log_emission[:, ot] if 0 <= ot < self.num_observations else np.full(N, neginf)
            for j in range(N):
                # compute delta[t-1, i] + logA[i,j] for all i
                scores = delta[t-1, :] + self.log_transition[:, j]
                i_max = np.argmax(scores)
                delta[t, j] = scores[i_max] + emit_t[j]
                psi[t, j] = i_max

        # backtrack
        states = [0] * T
        states[T-1] = int(np.argmax(delta[T-1, :]))
        for t in range(T-2, -1, -1):
            states[t] = int(psi[t+1, states[t+1]])
        return states

# @register_model("PomegranateHMMArabicModel")
# class PomegranateHMMArabicModel:
    # def __init__(self, char2id, diac2id, pad_id=None):
    #     self.char2id = char2id
    #     self.id2char = {v: k for k, v in char2id.items()}
    #     self.diac2id = diac2id
    #     self.id2diac = {v: k for k, v in diac2id.items()}
    #     self.pad_id = pad_id
    #     self.model = None

    # def _build_states(self, observations, labels):
    #     """
    #     Create pomegranate State objects for all diacritic (labels) with
    #     emission probabilities learned from training data.
    #     """
    #     # count emissions: state -> {obs: count}
    #     emission_counts = defaultdict(lambda: defaultdict(int))
    #     for obs_seq, state_seq in zip(observations, labels):
    #         for o, s in zip(obs_seq, state_seq):
    #             emission_counts[s][o] += 1

    #     states = {}
    #     for state, obs_dict in emission_counts.items():
    #         # normalize counts to probabilities
    #         total = sum(obs_dict.values())
    #         dist = {k: v / total for k, v in obs_dict.items()}
    #         states[state] = State(DiscreteDistribution(dist), name=state)
    #     return states

    # def fit(self, sequences, labels):
    #     """
    #     sequences: list of list of characters (observations)
    #     labels: list of list of diacritic labels (states)
    #     """
    #     # Build states with emission distributions
    #     states = self._build_states(sequences, labels)

    #     # Initialize model
    #     model = DenseHMM('ArabicDiacritizer')

    #     # Add states
    #     for state_obj in states.values():
    #         model.add_state(state_obj)

    #     # Start probabilities
    #     start_counts = {}
    #     for label_seq in labels:
    #         if label_seq:
    #             start_counts[label_seq[0]] = start_counts.get(label_seq[0], 0) + 1
    #     total_starts = sum(start_counts.values())
    #     for state_name, count in start_counts.items():
    #         model.add_transition(model.start, states[state_name], count / total_starts)

    #     # Transition probabilities
    #     from collections import defaultdict
    #     trans_counts = defaultdict(lambda: defaultdict(int))
    #     for label_seq in labels:
    #         for i in range(len(label_seq)-1):
    #             trans_counts[label_seq[i]][label_seq[i+1]] += 1

    #     # add transitions
    #     for from_state, to_dict in trans_counts.items():
    #         total = sum(to_dict.values())
    #         for to_state, cnt in to_dict.items():
    #             model.add_transition(states[from_state], states[to_state], cnt / total)

    #     # End transitions
    #     end_counts = defaultdict(int)
    #     for label_seq in labels:
    #         if label_seq:
    #             end_counts[label_seq[-1]] += 1
    #     total_ends = sum(end_counts.values())
    #     for state_name, cnt in end_counts.items():
    #         model.add_transition(states[state_name], model.end, cnt / total_ends)

    #     model.bake()
    #     self.model = model
    #     return self
    
    # def viterbi_decode(self, obs_seq):
    #     """
    #     obs_seq: list of characters (undiacritized)
    #     returns: list of predicted diacritics (strings)
    #     """
    #     if self.model is None:
    #         raise ValueError("Model not trained yet")

    #     logp, path = self.model.viterbi(obs_seq)
    #     # exclude start/end states
    #     decoded = [state[1].name for state in path[1:-1]]
    #     return decoded

    # def decode_batch(self, obs_seqs):
        return [self.viterbi_decode(seq) for seq in obs_seqs]