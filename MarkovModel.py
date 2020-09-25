import math
import operator

# rules for input:
# The hidden layer must have a length greater than the emission layer by a size of 2. This is to account for the start
# end symbol. The start and end symbol cannot exist anywhere inside the hidden layer except at the start and the end.
# The start symbol is denoted by "s" and the end symbol by "e".
# sample:
#
#  Hidden :    sffttffe
# Emission:     135625
#

class MarkovModel:

    all_emitted_characters = {}
    all_hidden_characters = {}

    emission_matrix = {}

    transition_matrix = {}

    symbol_count_with_smoothing = {}
    symbol_count_no_smoothing = {}

    def __init__(self, df):
        '''
        :param df: dataframe of transaction text in row[0] and hidden state text in row[1]
        builds an emission and transition matrix from the df
        '''
        for index, row in df.iterrows():
            emitted = row[0]
            hidden = row[1]
            if len(emitted) + 2 != len(hidden):
                raise Exception("Hidden layer must have a length greater than the emission layer by a size of 2")
            for i in range(len(emitted)):
                self.all_emitted_characters[emitted[i]] = 0
                if hidden[i+1] == 's' or hidden[i+1] == 'e':
                    raise Exception("Hidden layer cannot have the symbol 's' or 'e' except at the start and end")
                else:
                    self.all_hidden_characters[hidden[i+1]] = 0

            if hidden[0] != 's':
                raise Exception("Start of hidden layer must be 's'")

            if hidden[len(hidden)-1] != 'e':
                raise Exception("Start of hidden layer must be 's'")

        self.all_hidden_characters['s'] = 0
        self.all_hidden_characters['e'] = 0

        self.symbol_count_with_smoothing = self.all_hidden_characters.copy()
        self.symbol_count_no_smoothing = self.all_hidden_characters.copy()

        for symbol in self.all_hidden_characters:
            self.emission_matrix[symbol] = self.all_emitted_characters.copy()
            self.transition_matrix[symbol] = self.all_hidden_characters.copy()

        for symbol in self.emission_matrix:
            if symbol == 's':
                self.emission_matrix[symbol]['s'] = 1
            elif symbol == 'e':
                self.emission_matrix[symbol]['e'] = 1
            else:
                for character in self.emission_matrix[symbol]:
                    if character != 's' or character != 'e':
                        self.emission_matrix[symbol][character] = 1
                        self.symbol_count_with_smoothing [symbol] += 1
        for index, row in df.iterrows():
            self.build_matrix(row[0], row[1])
        self.get_probabilities()

    def build_matrix(self, transaction_txt, transaction_hidden):
        '''
        :param transaction_txt: a single transaction text
        :param transaction_hidden: hidden state corresponding to transaction text
        :return: none
        '''
        transaction_txt_len = len(transaction_txt)
        current_symbol = transaction_hidden[0]
        for i in range(1, transaction_txt_len+1):
            next_symbol = transaction_hidden[i]
            self.transition_matrix[current_symbol][next_symbol] += 1
            self.emission_matrix[next_symbol][transaction_txt[i-1]] += 1
            self.symbol_count_with_smoothing[next_symbol] += 1
            self.symbol_count_no_smoothing[next_symbol] += 1
            current_symbol = next_symbol
        self.transition_matrix[current_symbol]['e'] += 1
        self.symbol_count_no_smoothing['s'] += 1

    def get_probabilities(self):
        '''
        :return: Converts the count of characters/symbols in all matrices to probabilities
        '''
        important_symbols = ['t', 'f']
        for symbol in important_symbols:
            for next_symbol in self.transition_matrix[symbol]:
                self.transition_matrix[symbol][next_symbol] /= self.symbol_count_no_smoothing[symbol]
            for emission_char in self.emission_matrix[symbol]:
                self.emission_matrix[symbol][emission_char] /= self.symbol_count_with_smoothing[symbol]
        for start_transition in self.transition_matrix['s']:
            self.transition_matrix['s'][start_transition] /= self.symbol_count_no_smoothing['s']

    @staticmethod
    def predict_by_viterbi(transaction_txt, emission_matrix, transition_matrix, to_consider=1, constraints = None):
        '''
        :param transaction_txt: text to be examined
        :return: Returns the most likely sequence of states
        '''
        possible_states = transition_matrix.copy()
        del possible_states['s']
        del possible_states['e']
        best_previous = [{}]
        prob_previous = [{}]
        back_tracking = []
        for state in possible_states:
            prob_previous[0][state] = math.log(
                transition_matrix['s'][state] * emission_matrix[state][transaction_txt[0]])
        for i in range(1, len(transaction_txt)):
            prob_at_this_point = {}
            best_at_this_point = {}
            cur_emission = transaction_txt[i]
            previous = prob_previous[i - 1]
            for state_cur in possible_states:
                mx = -math.inf
                for state_prev in possible_states:
                    cur_mx = previous[state_prev] + math.log(
                        emission_matrix[state_cur][cur_emission] * transition_matrix[state_prev][state_cur])
                    if cur_mx > mx:
                        best_prev = state_prev
                        mx = cur_mx
                best_at_this_point[state_cur] = best_prev
                prob_at_this_point[state_cur] = mx
            prob_previous.append(prob_at_this_point)
            best_previous.append(best_at_this_point)

        # Special consideration for the end of the string
        last_mx = -math.inf
        for end_states in possible_states:
            prob_at_end = prob_at_this_point[end_states] + math.log(transition_matrix[end_states]['e'])
            if prob_at_end > last_mx:
                last_state = end_states
                last_mx = prob_at_end
        back_tracking.append(last_state)

        # Backtracking to find best chain
        for j in range(len(transaction_txt) - 1, 0, -1):
            prev_step = prob_previous[j]
            back_tracking.append(max(prev_step.items(), key=operator.itemgetter(1))[0])

        result = ""
        result.join(back_tracking)

        return result.join(back_tracking)

    @staticmethod
    def quick_partition(lst, start, end, size):
        '''
        :param lst: list to be partitioned
        :param start: first index of subsection to be partitioned
        :param end: last index of subsection to be partitioned
        :param size: will partition the largest (size) elements the start of the list, reducing complexity
        :return: A semi sorted list where the largest (size) elements are at the head of the list
        '''
        if start < end:
            pivot = list(lst[start].values())[0]
            counter = start
            for i in range(start + 1, end):
                if list(lst[i].values())[0] > pivot:
                    temp = lst[counter]
                    lst[counter] = lst[i]
                    lst[i] = temp
                    counter += 1
            if counter == start:
                MarkovModel.quick_partition(lst, start + 1, end, size - 1)
            elif counter == end - 1:
                MarkovModel.quick_partition(lst, start, end - 1, size)
            elif counter < size:
                MarkovModel.quick_partition(lst, counter, end, size - counter)
            elif counter > size:
                MarkovModel.quick_partition(lst, start, counter, size)

    @staticmethod
    def quick_sort(lst, start, end):
        '''
        :param lst: list to be sorted
        :param start: first index of subsection to be sorted
        :param end: last index of subsection to be sorted
        :return: in place sorted list done by quicksort
        '''
        if end - start > 1:
            pivot = list(lst[start].values())[0]
            counter = start
            for i in range(start + 1, end):
                if list(lst[i].values())[0] > pivot:
                    temp = lst[counter]
                    lst[counter] = lst[i]
                    lst[i] = temp
                    counter += 1
            if counter == start:
                MarkovModel.quick_sort(lst, start + 1, end)
            elif counter == end - 1:
                MarkovModel.quick_sort(lst, start, end - 1)
            else:
                MarkovModel.quick_sort(lst, counter, end)
                MarkovModel.quick_sort(lst, start, counter)

    def satisfies_constraints(self, text_to_be_examined):
        '''
        :param text_to_be_examined:
        :return: boolean if text satisfies constraints
        Current the constraints are:
            - if there are f symbols, they must be between max_f and min_f
            - the number of t symbols must be greater than min_t
            - if there are t symbols, they must be contiguous


        If all other chains fail, the first sequence is selected
        '''
        first_condition = self.min_f <= text_to_be_examined.count('f') <= self.max_f

        secnd_condition = self.min_t <= text_to_be_examined.count('t')

        third_condition = True
        seen_t = False
        past_t = False
        for i in range(len(text_to_be_examined)):
            char = text_to_be_examined[i]
            if char == 't':
                if seen_t:
                    if past_t:
                        third_condition = False
                        break
                else:
                    seen_t = True
            else:
                if seen_t:
                    past_t = True

        return first_condition and secnd_condition and third_condition

    @staticmethod
    def predict_by_rules(transaction_txt,emission_matrix, transition_matrix, to_consider, constraints):
        '''
        :param transaction_txt: text to be examined
        :param to_consider: number of sequences we will examine to see if they satisfy the constraint
        :return: Returns the most likely sequence of states that satisfies some constraints
        '''
        possible_states = emission_matrix.copy()
        txt_len = len(transaction_txt)
        del possible_states['s']
        del possible_states['e']
        queue = []
        left = {'t': math.log10(transition_matrix['s']['t'] * emission_matrix['t'][transaction_txt[0]])}
        right = {'f': math.log10(transition_matrix['s']['f'] * emission_matrix['f'][transaction_txt[0]])}
        queue.append(left)
        queue.append(right)

        special_transition_dict = {txt_len-1: [transition_matrix['t']['e'], transition_matrix['f']['e']]}
        # Using -1 to demarcate a new level in the tree
        queue.append(-1)
        i = 1

        while i < txt_len - 1:
            if queue[0] == -1:
                queue.pop(0)
                if len(queue) > to_consider:
                    MarkovModel.quick_partition(queue, 0, len(queue), to_consider)
                    queue = queue[:to_consider]
                queue.append(-1)
                i += 1
            else:
                cur_node = queue[0]
                prev_chain = list(cur_node.keys())[0]
                prev_symbol = prev_chain[i-1]
                prev_prob = cur_node[prev_chain]
                cur_emission = transaction_txt[i]
                left = {prev_chain + 't': prev_prob + math.log10(
                    transition_matrix[prev_symbol]['t']
                    * emission_matrix['t'][cur_emission]
                    * special_transition_dict.get(i, [1, 1])[0])}
                right = {prev_chain + 'f': prev_prob + math.log10(
                    transition_matrix[prev_symbol]['f']
                    * emission_matrix['f'][cur_emission]
                    * special_transition_dict.get(i, [1, 1])[1])}
                queue.append(left)
                queue.append(right)
                queue.pop(0)
        queue.pop(len(queue) - 1)
        size_of_leaves = len(queue)
        queue.pop(size_of_leaves - 1)
        if 0 < to_consider < len(queue):
            MarkovModel.quick_partition(queue, 0, size_of_leaves-1, to_consider)
            MarkovModel.quick_sort(queue, 0, to_consider)
            queue = queue[:to_consider]
        else:
            MarkovModel.quick_sort(queue, 0, size_of_leaves-1)
        for possible_chain in queue:
            current_chain = list(possible_chain.keys())[0]
            satisfies_constrains = True
            for fun in constraints:
                satisfies_constrains = satisfies_constrains and bool(fun(current_chain))
            if satisfies_constrains:
                return current_chain

        # If none satisfies the constraints, return the most likely sequence which is the first in the queue
        return list(queue[0].keys())[0]


    def predict(self, transaction_txt, method = predict_by_viterbi, to_consider = 10, constraints = []):
        '''
        :param transaction_txt: text to be examined
        :param method: method by which to extract best sequence
        :param to_consider: How many sequences are considered for predict_by_rules.
        :param constraints: list of user defined functions that return a boolean which the predicted
        hidden layer must statisfy
        possible methods are [predict_by_rules, predict_by_viterbi]
        predict_by_viterbi returns the best possible sequence
        predict_by_rules returns the best possible sequence that satisfies some constraints
        :return: The predicted hidden sequence
        '''
        return method.__func__(transaction_txt, self.emission_matrix, self.transition_matrix, to_consider, constraints)
       




        









