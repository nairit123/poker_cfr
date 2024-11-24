#!/usr/bin/env python3

import os
import argparse
import json

###############################################################################
# The next functions are already implemented for your convenience
#
# In all the functions in this stub file, `game` is the parsed input game json
# file, whereas `tfsdp` is either `game["decision_problem_pl1"]` or
# `game["decision_problem_pl2"]`.
#
# See the homework handout for a description of each field.


def get_sequence_set(tfsdp):
    """Returns a set of all sequences in the given tree-form sequential decision
    process (TFSDP)"""

    sequences = set()
    for node in tfsdp:
        if node["type"] == "decision":
            for action in node["actions"]:
                sequences.add((node["id"], action))
    return sequences


def is_valid_RSigma_vector(tfsdp, obj):
    """Checks that the given object is a dictionary keyed on the set of sequences
    of the given tree-form sequential decision process (TFSDP)"""

    sequence_set = get_sequence_set(tfsdp)
    return isinstance(obj, dict) and obj.keys() == sequence_set


def myassert(tfsdp, obj):
    """Checks whether the given object `obj` represents a valid sequence-form
    strategy vector for the given tree-form sequential decision process
    (TFSDP)"""

    if not is_valid_RSigma_vector(tfsdp, obj):
        print("The sequence-form strategy should be a dictionary with key set equal to the set of sequences in the game")
        os.exit(1)
    for node in tfsdp:
        if node["type"] == "decision":
            parent_reach = 1.0
            if node["parent_sequence"] is not None:
                parent_reach = obj[node["parent_sequence"]]
            if abs(sum([obj[(node["id"], action)] for action in node["actions"]]) - parent_reach) > 1e-2:
                print(
                    "At node ID %s the sum of the child sequences is not equal to the parent sequence", node["id"])
                print(f"Got sum {sum([obj[(node['id'], action)] for action in node['actions']])}, expected sum {parent_reach}")


def assert_is_valid_sf_strategy(tfsdp, obj):
    """Checks whether the given object `obj` represents a valid sequence-form
    strategy vector for the given tree-form sequential decision process
    (TFSDP)"""

    if not is_valid_RSigma_vector(tfsdp, obj):
        print("The sequence-form strategy should be a dictionary with key set equal to the set of sequences in the game")
        os.exit(1)
    for node in tfsdp:
        if node["type"] == "decision":
            parent_reach = 1.0
            if node["parent_sequence"] is not None:
                parent_reach = obj[node["parent_sequence"]]
            if abs(sum([obj[(node["id"], action)] for action in node["actions"]]) - parent_reach) > 1e-3:
                #print("At node ID %s the sum of the child sequences is not equal to the parent sequence", node["id"])
                pass


def best_response_value(tfsdp, utility):
    """Computes the value of max_{x in Q} x^T utility, where Q is the
    sequence-form polytope for the given tree-form sequential decision
    process (TFSDP)"""

    assert is_valid_RSigma_vector(tfsdp, utility)

    utility_ = utility.copy()
    utility_[None] = 0.0
    for node in tfsdp[::-1]:
        if node["type"] == "decision":
            max_ev = max([utility_[(node["id"], action)]
                         for action in node["actions"]])
            utility_[node["parent_sequence"]] += max_ev
    return utility_[None]


def compute_utility_vector_pl1(game, sf_strategy_pl2):
    """Returns A * y, where A is the payoff matrix of the game and y is
    the given strategy for Player 2"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl2"], sf_strategy_pl2)

    sequence_set = get_sequence_set(game["decision_problem_pl1"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl1"]] += entry["value"] * \
            sf_strategy_pl2[entry["sequence_pl2"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl1"], utility)
    return utility


def compute_utility_vector_pl2(game, sf_strategy_pl1):
    """Returns -A^transpose * x, where A is the payoff matrix of the
    game and x is the given strategy for Player 1"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl1"], sf_strategy_pl1)

    sequence_set = get_sequence_set(game["decision_problem_pl2"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl2"]] -= entry["value"] * \
            sf_strategy_pl1[entry["sequence_pl1"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl2"], utility)
    return utility


def gap(game, sf_strategy_pl1, sf_strategy_pl2):
    """Computes the saddle point gap of the given sequence-form strategies
    for the players"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(
        game["decision_problem_pl2"], sf_strategy_pl2)

    utility_pl1 = compute_utility_vector_pl1(game, sf_strategy_pl2)
    utility_pl2 = compute_utility_vector_pl2(game, sf_strategy_pl1)

    return (best_response_value(game["decision_problem_pl1"], utility_pl1)
            + best_response_value(game["decision_problem_pl2"], utility_pl2))


###########################################################################
# Starting from here, you should fill in the implementation of the
# different functions


def expected_utility_pl1(game, sf_strategy_pl1, sf_strategy_pl2):
    """Returns the expected utility for Player 1 in the game, when the two
    players play according to the given strategies"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(
        game["decision_problem_pl2"], sf_strategy_pl2)
    # FINISH

    payoff = 0
    for entry in game["utility_pl1"]:
        payoff += sf_strategy_pl1[entry["sequence_pl1"]] * sf_strategy_pl2[entry["sequence_pl2"]] * entry["value"]
    return payoff


def uniform_sf_strategy(tfsdp):
    """Returns the uniform sequence-form strategy for the given tree-form
    sequential decision process"""

    strategy = {}
    sequence_set = get_sequence_set(tfsdp)
    for sequence in sequence_set:
        strategy[sequence] = -1.0

    def compute(seq):
        if strategy[seq] > -0.5:
           return strategy[seq]
        decision_point = seq[0]
        for node in tfsdp:
            if node["id"] == decision_point and node["type"] == "decision":
                n = len(node["actions"])
                cur_node = node
        parent_reach = 1
        if cur_node["parent_sequence"] is not None:
            parent_reach = compute(cur_node["parent_sequence"])
        for action in cur_node["actions"]:
            strategy[(cur_node["id"], action)] = parent_reach * (1.0/n)
        return strategy[seq]

    for node in tfsdp:
        if node["type"] == "decision":
            for action in node["actions"]:
                compute((node["id"], action))

    return strategy

class RegretMatching(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.cum_regret  = dict()
        for action in action_set:
            self.cum_regret[action] = 0
        self.last_strat = dict()

    def next_strategy(self):
        # FINISH
        # You might want to return a dictionary mapping each action in
        # `self.action_set` to the probability of picking that action

        ret = dict()

        normalizer = 0
        for action in self.action_set:
            normalizer += max(0, self.cum_regret[action])

        if normalizer == 0:
            for action in self.action_set:
                ret[action] = 1.0/len(self.action_set)
        else:
            for action in self.action_set:
                ret[action] = max(0, self.cum_regret[action])/normalizer
        self.last_strat = ret
        return ret

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        strat_util = 0 # <u, x>
        for action in self.action_set:
            strat_util += (self.last_strat[action] * utility[action])

        for action in self.action_set:
            self.cum_regret[action] += (utility[action] - strat_util)

# strong - not proven
class RegretMatchingDom(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.cum_regret  = dict()
        for action in action_set:
            self.cum_regret[action] = 0
        self.last_strat = dict()

        self.graph = dict() # graph of dominating edges
        for action in action_set:
            self.graph[action] = [] # linked list
        self.graph_empty = True # have not initialized graph yet
        self.f = dict()
        for action in action_set:
            self.f[action] = action

        self.iter = 0
        self.c = 500
        self.C = 15

    def rec_help(self, action):
        if len(self.graph[action]) == 0:
            return action
        return self.rec_help(self.graph[action][0])

    def update_f(self):
        """
        print("Graph Representation (Adjacency List):")
        for vertex in self.action_set:
            neighbors = self.graph.get(vertex, [])
            print(f"{vertex} -> {', '.join(map(str, neighbors)) if neighbors else 'No edges'}")
        """
        for action in self.action_set:
            # go as far as we can recursively
            self.f[action] = self.rec_help(action)


    def next_strategy(self):
        # FINISH
        # You might want to return a dictionary mapping each action in
        # `self.action_set` to the probability of picking that action

        self.iter += 1
        ret = dict()

        normalizer = 0
        for action in self.action_set:
            normalizer += max(0, self.cum_regret[action])

        if normalizer == 0:
            for action in self.action_set:
                ret[action] = 1.0/len(self.action_set)
        else:
            for action in self.action_set:
                ret[action] = max(0, self.cum_regret[action])/normalizer

        self.last_strat = ret

        if self.iter >= self.c + self.C:
            dom_ret = dict()
            for action in self.action_set:
                dom_ret[action] = 0

            for action in self.action_set:
                dom_ret[self.f[action]] += ret[action]
                if self.f[action] != action and ret[action] > 0:
                    print("moved mass")

            return dom_ret
        else:
            return ret

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set


        strat_util = 0 # <u, x> where x is last strategy
        for action in self.action_set:
            strat_util += (self.last_strat[action] * utility[action])

        for action in self.action_set:
            self.cum_regret[action] += (utility[action] - strat_util)

        # if graph is empty, build it and modify f
        # later, when we wait before building, modify condition to if graph empty and iter big enough

        if self.iter >= self.c:
            if self.graph_empty:
                for action in self.action_set:
                    for action2 in self.action_set:
                        if action != action2:
                            if utility[action] < utility[action2]:
                                self.graph[action].append(action2)
                self.update_f()
                self.graph_empty = False
            else:
                for action in self.action_set:
                    # delete edges (action, to) if u[a] > u[t]
                    for i in range(len(self.graph[action]) - 1, -1, -1):
                        to = self.graph[action][i]
                        if utility[action] > utility[to]:
                            del self.graph[action][i]
                self.update_f()

        # if graph nonempty, remove edges as needed and modify f

class RegretMatchingPlus(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.cum_regret  = dict()
        for action in action_set:
            self.cum_regret[action] = 0
        self.last_strat = dict()

    def next_strategy(self):
        # FINISH
        # You might want to return a dictionary mapping each action in
        # `self.action_set` to the probability of picking that action

        ret = dict()

        normalizer = 0
        for action in self.action_set:
            normalizer += max(0, self.cum_regret[action])

        if normalizer == 0:
            for action in self.action_set:
                ret[action] = 1.0/len(self.action_set)
        else:
            for action in self.action_set:
                ret[action] = max(0, self.cum_regret[action])/normalizer
        self.last_strat = ret
        return ret

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        strat_util = 0 # <u, x>
        for action in self.action_set:
            strat_util += (self.last_strat[action] * utility[action])

        for action in self.action_set:
            self.cum_regret[action] += (utility[action] - strat_util)

        for action in self.action_set:
            self.cum_regret[action] = max(self.cum_regret[action], 0)

class RegretMatchingOptimisticPlus(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.cum_regret  = dict()
        self.lastg = dict()
        for action in action_set:
            self.cum_regret[action] = 0
            self.lastg[action] = 0
        self.last_strat = dict()

    def next_strategy(self):
        # FINISH
        # You might want to return a dictionary mapping each action in
        # `self.action_set` to the probability of picking that action


        for action in self.action_set:
            self.cum_regret[action] += self.lastg[action]
            self.cum_regret[action] = max(self.cum_regret[action], 0)

        ret = dict()

        normalizer = 0
        for action in self.action_set:
            normalizer += max(0, self.cum_regret[action])

        if normalizer < 0.001:
            for action in self.action_set:
                ret[action] = 1.0/len(self.action_set)
        else:
            for action in self.action_set:
                ret[action] = max(0, self.cum_regret[action])/normalizer
        self.last_strat = ret
        return ret

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        strat_util = 0 # <u, x>
        for action in self.action_set:
            strat_util += (self.last_strat[action] * utility[action])

        for action in self.action_set:
            self.cum_regret[action] += (utility[action] - strat_util)
            self.lastg[action] = (utility[action] - strat_util)
            self.cum_regret[action] = max(self.cum_regret[action], 0)

class RegretMatchingDiscount(object):
    def __init__(self, action_set):
        self.alpha = 1.5
        self.beta = 0
        self.t = 1
        self.action_set = set(action_set)
        self.cum_regret  = dict()
        for action in action_set:
            self.cum_regret[action] = 0
        self.last_strat = dict()

    def next_strategy(self):
        # FINISH
        # You might want to return a dictionary mapping each action in
        # `self.action_set` to the probability of picking that action

        ret = dict()

        normalizer = 0
        for action in self.action_set:
            normalizer += max(0, self.cum_regret[action])

        if normalizer == 0:
            for action in self.action_set:
                ret[action] = 1.0/len(self.action_set)
        else:
            for action in self.action_set:
                ret[action] = max(0, self.cum_regret[action])/normalizer
        self.last_strat = ret
        return ret

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set
        t = self.t
        beta = self.beta
        alpha=self.alpha
        strat_util = 0 # <u, x>
        for action in self.action_set:
            strat_util += (self.last_strat[action] * utility[action])

        for action in self.action_set:
            if utility[action] - strat_util > 0:
                self.cum_regret[action] += (utility[action] - strat_util) * (t**alpha)/(t**alpha + 1)
            else:
                self.cum_regret[action] += (utility[action] - strat_util) * (t**beta)/(t**beta + 1)
            #self.cum_regret[action] = max(self.cum_regret[action], 0) - this does better

        self.t += 1

class Cfr(object):
    def __init__(self, tfsdp, rm_class=RegretMatching):
        self.tfsdp = tfsdp
        self.local_regret_minimizers = {}

        # For each decision point, we instantiate a local regret minimizer
        for node in tfsdp:
            if node["type"] == "decision":
                self.local_regret_minimizers[node["id"]] = rm_class(
                    node["actions"])

    def next_strategy(self):
        strategy = {}
        sequence_set = get_sequence_set(self.tfsdp)
        for sequence in sequence_set:
            strategy[sequence] = -1.0

        def compute(seq):
            if strategy[seq] > -0.5:
               return strategy[seq]
            decision_point = seq[0]
            for node in self.tfsdp:
                if node["id"] == decision_point and node["type"] == "decision":
                    n = len(node["actions"])
                    cur_node = node
            parent_reach = 1
            if cur_node["parent_sequence"] is not None:
                parent_reach = compute(cur_node["parent_sequence"])
            assert(cur_node["type"] == "decision")
            local_strat = self.local_regret_minimizers[cur_node["id"]].next_strategy()
            for action in cur_node["actions"]:
                strategy[(cur_node["id"], action)] = parent_reach * local_strat[action]
            return strategy[seq]

        for node in self.tfsdp:
            if node["type"] == "decision":
                for action in node["actions"]:
                    compute((node["id"], action))
        #assert_is_valid_sf_strategy(self.tfsdp, strategy)
        myassert(self.tfsdp, strategy)

        # return overall strategy, probability distribution at each decision point
        # map decision node id -> (map actions -> float)
        localmap = dict()
        for node in self.tfsdp:
            if node["type"] == "decision":
                localmap[node["id"]] = self.local_regret_minimizers[node["id"]].next_strategy()
        return strategy, localmap



    # utility: counterfactual utilities for each sequence
    def observe_utility(self, utility):
        # let each local regret minimizer observe utilities
        for node in self.tfsdp:
            if node["type"] == "decision":
                local_util = dict()
                for action in node["actions"]:
                    local_util[action] = utility[(node["id"], action)]
                self.local_regret_minimizers[node["id"]].observe_utility(local_util)

def add_strat(tt, strat1, strat2):
    assert(is_valid_RSigma_vector(tt, strat1))
    assert(is_valid_RSigma_vector(tt, strat2))
    ret = dict()
    for node in tt:
        if node["type"] == "decision":
            for action in node["actions"]:
                ret[(node["id"], action)] = strat1[(node["id"], action)] + strat2[(node["id"], action)]
    return ret

def div_strat(tt, strat1, r):
    assert(is_valid_RSigma_vector(tt, strat1))
    assert(r != 0)

    ret = dict()
    for node in tt:
        if node["type"] == "decision":
            for action in node["actions"]:
                ret[(node["id"], action)] = strat1[(node["id"], action)]/r
    return ret

def mul_strat(tt, strat1, r):
    assert(is_valid_RSigma_vector(tt, strat1))

    ret = dict()
    for node in tt:
        if node["type"] == "decision":
            for action in node["actions"]:
                ret[(node["id"], action)] = strat1[(node["id"], action)]*r
    return ret


def solve_problem_3_1(game):

    tfs_p1 = game["decision_problem_pl1"]
    tfs_p2 = game["decision_problem_pl2"]

    # map decision point ids to the full decision point nodes for each player
    mp1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            mp1[node["id"]] = node
    mp2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            mp2[node["id"]] = node

    p2_strat = uniform_sf_strategy(tfs_p2)

    Cfr_p1 = Cfr(tfs_p1)

    cum_strat = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat[(node["id"], action)] = 0

    vT_vals = []
    for i in range(1, 1001):
        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        myassert(tfs_p1, p1_strat)
        cum_strat = add_strat(tfs_p1, cum_strat, p1_strat)

        observed_util = dict() # for each terminal node, we go up the game tree
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util[(node["id"], action)] = 0
        # for each terminal node, compute w(z) := pi^t_2(z) * c(z) * u_1(z), multiply all ancestor sequences (J, a) of z by w(z) pi_1(z|J, a), and update util[J, a] by this value

        for term in game["utility_pl1"]:
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # decision point node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1
            observed_util[(cur_node["id"], cur_action)] += wz * p1hza
            while True:
                p1hza *=  localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]
                observed_util[(cur_node["id"], cur_action)] += wz * p1hza

        Cfr_p1.observe_utility(observed_util)

        avg_strat = div_strat(tfs_p1, cum_strat, i)
        myassert(tfs_p1, avg_strat)
        vT_vals.append(expected_utility_pl1(game, avg_strat, p2_strat))


    import matplotlib.pyplot as plt
    plt.plot(range(1, 1001), vT_vals)
    plt.xlabel('T')
    plt.ylabel('v^T')
    plt.title(f'Plot of v^T as a function of T')
    plt.show()



def solve_problem_3_2(game):
    tfs_p1 = game["decision_problem_pl1"]
    tfs_p2 = game["decision_problem_pl2"]

    # map decision point ids to the full decision point nodes for each player
    mp1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            mp1[node["id"]] = node
    mp2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            mp2[node["id"]] = node

    Cfr_p1 = Cfr(tfs_p1)
    Cfr_p2 = Cfr(tfs_p2)

    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts = []
    # x[0] and y[0] initialized
    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    for i in range(1, 1001):

        # observe utilities simultaneously
        observed_util_1 = dict() # for each terminal node, we go up the game tree
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        observed_util_2 = dict() # for each terminal node, we go up the game tree
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            # cfr contributions for terminal node `term`
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # mp maps node id to node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1 # pi_1(z|ha)
            observed_util_1[(cur_node["id"], cur_action)] += wz
            while True:
                # utility contribution to all ancestors
                p1hza *=  localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]

                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p1.observe_utility(observed_util_1)
        Cfr_p2.observe_utility(observed_util_2)

        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        p2_strat, localmap_2 = Cfr_p2.next_strategy()
        cum_strat_1 = add_strat(tfs_p1, cum_strat_1, p1_strat)
        cum_strat_2 = add_strat(tfs_p2, cum_strat_2, p2_strat)
        avg_strat_1 = div_strat(tfs_p1, cum_strat_1, i)
        avg_strat_2 = div_strat(tfs_p2, cum_strat_2, i)

        saddle_pts.append(gap(game, avg_strat_1, avg_strat_2))


    # sanity check
    print(f"EV of player 1 under average strategies: {expected_utility_pl1(game, avg_strat_1, avg_strat_2)}")
    print(f"Saddle point gap: {saddle_pts[-1]}")

    import matplotlib.pyplot as plt
    plt.plot(range(1, 1001), saddle_pts)
    plt.xlabel('T')
    plt.ylabel('saddle')
    plt.title(f'Plot of saddle as a function of T')
    #plt.yscale('log')  # Using a log scale for y-axis to handle large values
    plt.show()




def solve_problem_3_3(game):
    gamma = 1
    tfs_p1 = game["decision_problem_pl1"]
    tfs_p2 = game["decision_problem_pl2"]

    # map decision point ids to the full decision point nodes for each player
    mp1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            mp1[node["id"]] = node
    mp2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            mp2[node["id"]] = node

    Cfr_p1 = Cfr(tfs_p1)
    Cfr_p2 = Cfr(tfs_p2)

    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_cfr = []
    util_pl1_cfr = []

    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    for i in range(1, 1001):

        # observe utilities simultaneously
        observed_util_1 = dict() # for each terminal node, we go up the game tree
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        observed_util_2 = dict() # for each terminal node, we go up the game tree
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            # cfr contributions for terminal node `term`
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # mp maps node id to node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1 # pi_1(z|ha)
            observed_util_1[(cur_node["id"], cur_action)] += wz
            while True:
                # utility contribution to all ancestors
                p1hza *=  localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]

                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p1.observe_utility(observed_util_1)
        Cfr_p2.observe_utility(observed_util_2)

        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        p2_strat, localmap_2 = Cfr_p2.next_strategy()
        cum_strat_1 = add_strat(tfs_p1, cum_strat_1, p1_strat)
        cum_strat_2 = add_strat(tfs_p2, cum_strat_2, p2_strat)
        avg_strat_1 = div_strat(tfs_p1, cum_strat_1, i)
        avg_strat_2 = div_strat(tfs_p2, cum_strat_2, i)

        util_pl1_cfr.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))
        saddle_pts_cfr.append(gap(game, avg_strat_1, avg_strat_2))

    print(f"saddle point of CFR simultaneous: {saddle_pts_cfr[-1]}")

    Cfr_p1 = Cfr(tfs_p1, rm_class=RegretMatchingPlus)
    Cfr_p2 = Cfr(tfs_p2, rm_class=RegretMatchingPlus)



    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_cfrplus = []
    util_pl1_cfrplus = []

    # initial (uniform) strategies
    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    normalizer = 0
    for i in range(1, 1001):
        normalizer += i ** gamma
        # player 1 asks CFR for a new strat
        # player 1 observes utilities computed from player 2's last strat
        # player 2 asks CFR for a new strat
        # player 2 observes utilities computed from player 1's last strat


        observed_util_1 = dict()
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # decision point node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1
            observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza
            while True:
                p1hza *= localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza


        Cfr_p1.observe_utility(observed_util_1)

        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        multiplied_1 = mul_strat(tfs_p1, p1_strat, i ** gamma)
        cum_strat_1 = add_strat(tfs_p1, cum_strat_1, multiplied_1)



        observed_util_2 = dict()
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p2.observe_utility(observed_util_2)

        p2_strat, localmap_2 = Cfr_p2.next_strategy()
        multiplied_2 = mul_strat(tfs_p2, p2_strat, i**gamma)
        cum_strat_2 = add_strat(tfs_p2, cum_strat_2, multiplied_2)

        avg_strat_1 = div_strat(tfs_p1, cum_strat_1, normalizer)
        avg_strat_2 = div_strat(tfs_p2, cum_strat_2, normalizer)

        saddle_pts_cfrplus.append(gap(game, avg_strat_1, avg_strat_2))
        util_pl1_cfrplus.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))


    print(f"saddle point of CFR+: {saddle_pts_cfrplus[-1]}")
    Cfr_p1 = Cfr(tfs_p1, rm_class=RegretMatchingDiscount)
    Cfr_p2 = Cfr(tfs_p2, rm_class=RegretMatchingDiscount)

    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_dcfr = []
    util_pl1_dcfr = []

    # initial (uniform) strategies
    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    normalizer = 0
    for i in range(1, 1001):
        normalizer += i ** 2
        # player 1 asks CFR for a new strat
        # player 1 observes utilities computed from player 2's last strat
        # player 2 asks CFR for a new strat
        # player 2 observes utilities computed from player 1's last strat


        observed_util_1 = dict()
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # decision point node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1
            observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza
            while True:
                p1hza *= localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza


        Cfr_p1.observe_utility(observed_util_1)

        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        multiplied_1 = mul_strat(tfs_p1, p1_strat, i ** 2)
        cum_strat_1 = add_strat(tfs_p1, cum_strat_1, multiplied_1)



        observed_util_2 = dict()
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p2.observe_utility(observed_util_2)

        p2_strat, localmap_2 = Cfr_p2.next_strategy()
        multiplied_2 = mul_strat(tfs_p2, p2_strat, i ** 2)
        cum_strat_2 = add_strat(tfs_p2, cum_strat_2, multiplied_2)

        avg_strat_1 = div_strat(tfs_p1, cum_strat_1, normalizer)
        avg_strat_2 = div_strat(tfs_p2, cum_strat_2, normalizer)

        saddle_pts_dcfr.append(gap(game, avg_strat_1, avg_strat_2))
        util_pl1_dcfr.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))


    print(f"saddle point of DCFR: {saddle_pts_dcfr[-1]}")

    Cfr_p1 = Cfr(tfs_p1, rm_class=RegretMatchingOptimisticPlus)
    Cfr_p2 = Cfr(tfs_p2, rm_class=RegretMatchingOptimisticPlus)

    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_pcfr1 = []
    util_pl1_pcfr1 = []

    # initial (uniform) strategies
    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    normalizer = 0
    for i in range(1, 1001):
        normalizer += i ** 2
        # player 1 asks CFR for a new strat
        # player 1 observes utilities computed from player 2's last strat
        # player 2 asks CFR for a new strat
        # player 2 observes utilities computed from player 1's last strat


        observed_util_1 = dict()
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # decision point node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1
            observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza
            while True:
                p1hza *= localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza


        Cfr_p1.observe_utility(observed_util_1)

        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        multiplied_1 = mul_strat(tfs_p1, p1_strat, i ** 2)
        cum_strat_1 = add_strat(tfs_p1, cum_strat_1, multiplied_1)


        observed_util_2 = dict()
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p2.observe_utility(observed_util_2)

        p2_strat, localmap_2 = Cfr_p2.next_strategy()
        multiplied_2 = mul_strat(tfs_p2, p2_strat, i ** 2)
        cum_strat_2 = add_strat(tfs_p2, cum_strat_2, multiplied_2)

        avg_strat_1 = div_strat(tfs_p1, cum_strat_1, normalizer)
        avg_strat_2 = div_strat(tfs_p2, cum_strat_2, normalizer)

        saddle_pts_pcfr1.append(gap(game, avg_strat_1, avg_strat_2))
        util_pl1_pcfr1.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))

    print(f"saddle point of PCFR1: {saddle_pts_pcfr1[-1]}")

    Cfr_p1 = Cfr(tfs_p1, rm_class=RegretMatchingOptimisticPlus)
    Cfr_p2 = Cfr(tfs_p2, rm_class=RegretMatchingOptimisticPlus)

    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_pcfr2 = []
    util_pl1_pcfr2 = []

    # initial (uniform) strategies
    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    normalizer = 0
    for i in range(1, 1001):
        normalizer += i ** 2
        # player 1 asks CFR for a new strat
        # player 1 observes utilities computed from player 2's last strat
        # player 2 asks CFR for a new strat
        # player 2 observes utilities computed from player 1's last strat


        observed_util_1 = dict()
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # decision point node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1
            observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza
            while True:
                p1hza *= localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza


        Cfr_p1.observe_utility(observed_util_1)
        p1_strat, localmap_1 = Cfr_p1.next_strategy()


        observed_util_2 = dict()
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p2.observe_utility(observed_util_2)
        p2_strat, localmap_2 = Cfr_p2.next_strategy()

        avg_strat_1 = p1_strat
        avg_strat_2 = p2_strat

        saddle_pts_pcfr2.append(gap(game, avg_strat_1, avg_strat_2))
        util_pl1_pcfr2.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))

    print(f"saddle point of PCFR2: {saddle_pts_pcfr2[-1]}")

    import matplotlib.pyplot as plt
    plt.plot(range(1, 1001), saddle_pts_cfr, label="CFR")
    plt.plot(range(1, 1001), saddle_pts_cfrplus, label="CFR+")
    plt.plot(range(1, 1001), saddle_pts_dcfr, label="DCFR")
    plt.plot(range(1, 1001), saddle_pts_pcfr1, label="PCFR 1")
    plt.plot(range(1, 1001), saddle_pts_pcfr2, label="PCFR 2")
    plt.xlabel('T')
    plt.ylabel('saddle')
    plt.title(f'Plot of saddle as a function of T')
    plt.legend(loc="best")
    plt.show()


    plt.plot(range(1, 1001), util_pl1_cfr, label="CFR")
    plt.plot(range(1, 1001), util_pl1_cfrplus, label="CFR+")
    plt.plot(range(1, 1001), util_pl1_dcfr, label="DCFR")
    plt.plot(range(1, 1001), util_pl1_pcfr1, label="PCFR 1")
    plt.plot(range(1, 1001), util_pl1_pcfr2, label="PCFR 2")
    plt.xlabel('T')
    plt.ylabel('EV')
    plt.title(f'Plot of utilities as a function of T')
    plt.legend(loc="best")
    plt.show()


# problem 3.4
# This is an extension of the hw
# This does the same thing as problem 3.3, but every regret minimizer becomes "dominated"
# Specifically, we use our theorem to replace each regret minimizer R with regret minimizer R'
def solve_problem_3_4(game):
    gamma = 1
    tfs_p1 = game["decision_problem_pl1"]
    tfs_p2 = game["decision_problem_pl2"]

    # map decision point ids to the full decision point nodes for each player
    mp1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            mp1[node["id"]] = node
    mp2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            mp2[node["id"]] = node

    Cfr_p1 = Cfr(tfs_p1, rm_class=RegretMatchingDom)
    Cfr_p2 = Cfr(tfs_p2, rm_class=RegretMatchingDom)

    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_cfr = []
    util_pl1_cfr = []

    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    for i in range(1, 1001):

        # observe utilities simultaneously
        observed_util_1 = dict() # for each terminal node, we go up the game tree
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        observed_util_2 = dict() # for each terminal node, we go up the game tree
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            # cfr contributions for terminal node `term`
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # mp maps node id to node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1 # pi_1(z|ha)
            observed_util_1[(cur_node["id"], cur_action)] += wz
            while True:
                # utility contribution to all ancestors
                p1hza *=  localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]

                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p1.observe_utility(observed_util_1)
        Cfr_p2.observe_utility(observed_util_2)

        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        p2_strat, localmap_2 = Cfr_p2.next_strategy()
        cum_strat_1 = add_strat(tfs_p1, cum_strat_1, p1_strat)
        cum_strat_2 = add_strat(tfs_p2, cum_strat_2, p2_strat)
        avg_strat_1 = div_strat(tfs_p1, cum_strat_1, i)
        avg_strat_2 = div_strat(tfs_p2, cum_strat_2, i)

        util_pl1_cfr.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))
        saddle_pts_cfr.append(gap(game, avg_strat_1, avg_strat_2))

    print(f"saddle point of CFR simultaneous: {saddle_pts_cfr[-1]}")

    Cfr_p1 = Cfr(tfs_p1, rm_class=RegretMatchingPlus)
    Cfr_p2 = Cfr(tfs_p2, rm_class=RegretMatchingPlus)



    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_cfrplus = []
    util_pl1_cfrplus = []

    # initial (uniform) strategies
    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    normalizer = 0
    for i in range(1, 1001):
        normalizer += i ** gamma
        # player 1 asks CFR for a new strat
        # player 1 observes utilities computed from player 2's last strat
        # player 2 asks CFR for a new strat
        # player 2 observes utilities computed from player 1's last strat


        observed_util_1 = dict()
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # decision point node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1
            observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza
            while True:
                p1hza *= localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza


        Cfr_p1.observe_utility(observed_util_1)

        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        multiplied_1 = mul_strat(tfs_p1, p1_strat, i ** gamma)
        cum_strat_1 = add_strat(tfs_p1, cum_strat_1, multiplied_1)



        observed_util_2 = dict()
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p2.observe_utility(observed_util_2)

        p2_strat, localmap_2 = Cfr_p2.next_strategy()
        multiplied_2 = mul_strat(tfs_p2, p2_strat, i**gamma)
        cum_strat_2 = add_strat(tfs_p2, cum_strat_2, multiplied_2)

        avg_strat_1 = div_strat(tfs_p1, cum_strat_1, normalizer)
        avg_strat_2 = div_strat(tfs_p2, cum_strat_2, normalizer)

        saddle_pts_cfrplus.append(gap(game, avg_strat_1, avg_strat_2))
        util_pl1_cfrplus.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))


    print(f"saddle point of CFR+: {saddle_pts_cfrplus[-1]}")
    Cfr_p1 = Cfr(tfs_p1, rm_class=RegretMatchingDiscount)
    Cfr_p2 = Cfr(tfs_p2, rm_class=RegretMatchingDiscount)

    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_dcfr = []
    util_pl1_dcfr = []

    # initial (uniform) strategies
    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    normalizer = 0
    for i in range(1, 1001):
        normalizer += i ** 2
        # player 1 asks CFR for a new strat
        # player 1 observes utilities computed from player 2's last strat
        # player 2 asks CFR for a new strat
        # player 2 observes utilities computed from player 1's last strat


        observed_util_1 = dict()
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # decision point node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1
            observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza
            while True:
                p1hza *= localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza


        Cfr_p1.observe_utility(observed_util_1)

        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        multiplied_1 = mul_strat(tfs_p1, p1_strat, i ** 2)
        cum_strat_1 = add_strat(tfs_p1, cum_strat_1, multiplied_1)



        observed_util_2 = dict()
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p2.observe_utility(observed_util_2)

        p2_strat, localmap_2 = Cfr_p2.next_strategy()
        multiplied_2 = mul_strat(tfs_p2, p2_strat, i ** 2)
        cum_strat_2 = add_strat(tfs_p2, cum_strat_2, multiplied_2)

        avg_strat_1 = div_strat(tfs_p1, cum_strat_1, normalizer)
        avg_strat_2 = div_strat(tfs_p2, cum_strat_2, normalizer)

        saddle_pts_dcfr.append(gap(game, avg_strat_1, avg_strat_2))
        util_pl1_dcfr.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))


    print(f"saddle point of DCFR: {saddle_pts_dcfr[-1]}")

    Cfr_p1 = Cfr(tfs_p1, rm_class=RegretMatchingOptimisticPlus)
    Cfr_p2 = Cfr(tfs_p2, rm_class=RegretMatchingOptimisticPlus)

    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_pcfr1 = []
    util_pl1_pcfr1 = []

    # initial (uniform) strategies
    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    normalizer = 0
    for i in range(1, 1001):
        normalizer += i ** 2
        # player 1 asks CFR for a new strat
        # player 1 observes utilities computed from player 2's last strat
        # player 2 asks CFR for a new strat
        # player 2 observes utilities computed from player 1's last strat


        observed_util_1 = dict()
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # decision point node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1
            observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza
            while True:
                p1hza *= localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza


        Cfr_p1.observe_utility(observed_util_1)

        p1_strat, localmap_1 = Cfr_p1.next_strategy()
        multiplied_1 = mul_strat(tfs_p1, p1_strat, i ** 2)
        cum_strat_1 = add_strat(tfs_p1, cum_strat_1, multiplied_1)


        observed_util_2 = dict()
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p2.observe_utility(observed_util_2)

        p2_strat, localmap_2 = Cfr_p2.next_strategy()
        multiplied_2 = mul_strat(tfs_p2, p2_strat, i ** 2)
        cum_strat_2 = add_strat(tfs_p2, cum_strat_2, multiplied_2)

        avg_strat_1 = div_strat(tfs_p1, cum_strat_1, normalizer)
        avg_strat_2 = div_strat(tfs_p2, cum_strat_2, normalizer)

        saddle_pts_pcfr1.append(gap(game, avg_strat_1, avg_strat_2))
        util_pl1_pcfr1.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))

    print(f"saddle point of PCFR1: {saddle_pts_pcfr1[-1]}")

    Cfr_p1 = Cfr(tfs_p1, rm_class=RegretMatchingOptimisticPlus)
    Cfr_p2 = Cfr(tfs_p2, rm_class=RegretMatchingOptimisticPlus)

    cum_strat_1 = dict()
    for node in tfs_p1:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_1[(node["id"], action)] = 0

    cum_strat_2 = dict()
    for node in tfs_p2:
        if node["type"] == "decision":
            for action in node["actions"]:
                cum_strat_2[(node["id"], action)] = 0

    saddle_pts_pcfr2 = []
    util_pl1_pcfr2 = []

    # initial (uniform) strategies
    p1_strat, localmap_1 = Cfr_p1.next_strategy()
    p2_strat, localmap_2 = Cfr_p2.next_strategy()
    normalizer = 0
    for i in range(1, 1001):
        normalizer += i ** 2
        # player 1 asks CFR for a new strat
        # player 1 observes utilities computed from player 2's last strat
        # player 2 asks CFR for a new strat
        # player 2 observes utilities computed from player 1's last strat


        observed_util_1 = dict()
        for node in tfs_p1:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_1[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p2_strat[term["sequence_pl2"]] * term["value"]
            cur_node = mp1[term["sequence_pl1"][0]] # decision point node
            cur_action = term["sequence_pl1"][1]
            p1hza = 1
            observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza
            while True:
                p1hza *= localmap_1[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp1[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_1[(cur_node["id"], cur_action)] += wz * p1hza


        Cfr_p1.observe_utility(observed_util_1)
        p1_strat, localmap_1 = Cfr_p1.next_strategy()


        observed_util_2 = dict()
        for node in tfs_p2:
            if node["type"] == "decision":
                for action in node["actions"]:
                    observed_util_2[(node["id"], action)] = 0

        for term in game["utility_pl1"]:
            wz = p1_strat[term["sequence_pl1"]] * term["value"]
            cur_node = mp2[term["sequence_pl2"][0]] # decision point node
            cur_action = term["sequence_pl2"][1]
            p2hza = 1
            observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza
            while True:
                p2hza *= localmap_2[cur_node["id"]][cur_action]
                p_seq = cur_node["parent_sequence"]
                if p_seq is None:
                    break
                cur_node = mp2[p_seq[0]]
                cur_action = p_seq[1]
                observed_util_2[(cur_node["id"], cur_action)] -= wz * p2hza


        Cfr_p2.observe_utility(observed_util_2)
        p2_strat, localmap_2 = Cfr_p2.next_strategy()

        avg_strat_1 = p1_strat
        avg_strat_2 = p2_strat

        saddle_pts_pcfr2.append(gap(game, avg_strat_1, avg_strat_2))
        util_pl1_pcfr2.append(expected_utility_pl1(game, avg_strat_1, avg_strat_2))

    print(f"saddle point of PCFR2: {saddle_pts_pcfr2[-1]}")

    import matplotlib.pyplot as plt
    plt.plot(range(1, 1001), saddle_pts_cfr, label="CFR")
    plt.plot(range(1, 1001), saddle_pts_cfrplus, label="CFR+")
    plt.plot(range(1, 1001), saddle_pts_dcfr, label="DCFR")
    plt.plot(range(1, 1001), saddle_pts_pcfr1, label="PCFR 1")
    plt.plot(range(1, 1001), saddle_pts_pcfr2, label="PCFR 2")
    plt.xlabel('T')
    plt.ylabel('saddle')
    plt.title(f'Plot of saddle as a function of T')
    plt.legend(loc="best")
    plt.show()


    plt.plot(range(1, 1001), util_pl1_cfr, label="CFR")
    plt.plot(range(1, 1001), util_pl1_cfrplus, label="CFR+")
    plt.plot(range(1, 1001), util_pl1_dcfr, label="DCFR")
    plt.plot(range(1, 1001), util_pl1_pcfr1, label="PCFR 1")
    plt.plot(range(1, 1001), util_pl1_pcfr2, label="PCFR 2")
    plt.xlabel('T')
    plt.ylabel('EV')
    plt.title(f'Plot of utilities as a function of T')
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Problem 3 (CFR)')
    parser.add_argument("--game", help="Path to game file")
    parser.add_argument("--problem", choices=["3.1", "3.2", "3.3", "3.4"])

    args = parser.parse_args()
    print("Reading game path %s..." % args.game)

    game = json.load(open(args.game))

    # Convert all sequences from lists to tuples
    for tfsdp in [game["decision_problem_pl1"], game["decision_problem_pl2"]]:
        for node in tfsdp:
            if isinstance(node["parent_edge"], list):
                node["parent_edge"] = tuple(node["parent_edge"])
            if "parent_sequence" in node and isinstance(node["parent_sequence"], list):
                node["parent_sequence"] = tuple(node["parent_sequence"])
    for entry in game["utility_pl1"]:
        assert isinstance(entry["sequence_pl1"], list)
        assert isinstance(entry["sequence_pl2"], list)
        entry["sequence_pl1"] = tuple(entry["sequence_pl1"])
        entry["sequence_pl2"] = tuple(entry["sequence_pl2"])

    print("... done. Running code for Problem", args.problem)

    if args.problem == "3.1":
        solve_problem_3_1(game)
    elif args.problem == "3.2":
        solve_problem_3_2(game)
    elif args.problem == "3.3":
        solve_problem_3_3(game)
    else:
        assert args.problem == "3.4"
        solve_problem_3_4(game)
