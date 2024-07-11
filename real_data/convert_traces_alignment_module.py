import re

class UnrecognizedActionTypeException(Exception):
    pass
class UnrecognizedLineException(Exception):
    pass
class PlanAndTraceDoNotMatchException(Exception):
    pass
class NoPlanMatchedTheTraceException(Exception):
    pass
class EarlyGoalException(Exception):
    pass

MAIN_PATTERN = re.compile(r"\( (.+?) - (.+) - (.+?) \)", re.VERBOSE)
GOAL_PATTERN = re.compile(r"\( gotogoal - (.+?) \)", re.VERBOSE)

class Aligner:

    def __init__(self, plans_file, alphabet):
        self.plans_file = plans_file
        self.alphabet = alphabet
        self.alphabet_compat = [self._pred_common_denominator(s) for s in self.alphabet]
        self.reset_all_plans()
    
    def _pred_common_denominator(self, predicate):
        return predicate.replace("_", "")
    
    def reset_all_plans(self):
        with open(self.plans_file) as f:
            content = f.read()
            self.all_plans = re.split("=+", content)
        self.all_plans = [plan for plan in self.all_plans if len(plan.strip())>0]
    
    # get plan
    def __getitem__(self, i):
        return self.all_plans[i].strip().splitlines()
    
    def __len__(self):
        return len(self.all_plans)

    def align(self, trace_list, trace_num):
        plan = self[trace_num]
        return self._align_to_plan(trace_list, plan)

    def align_BRUTEFORCE(self, trace_list, _trace_num_for_debug=None):
        matched = set()
        for i, plan in enumerate(self):
            try:
                new_trace = self._align_to_plan(trace_list, plan)
                # self.all_plans.pop(i)
                matched.add(i)
            except PlanAndTraceDoNotMatchException as e:
                # print("Mismatch", i, "on ti =", e.args)
                pass
            except EarlyGoalException:
                print("Early goal", i)
        if len(matched) > 0:
            print("Got", len(matched), "matches:", matched)
            return new_trace
        else:
            raise NoPlanMatchedTheTraceException()

    def _conform_predicate(self, predicate):
        if predicate in self.alphabet:
            return predicate
        # raises ValueError in the unlikely case that the predicate is not in the aplhabet at all
        i = self.alphabet_compat.index(self._pred_common_denominator(predicate))
        return self.alphabet[i]

    def _align_to_plan(self, trace_list, plan):
        # So this is NOT in place
        # It's a list of strings, so no deepcopy is necessary
        trace_list = trace_list.copy()
        # parse plan and align trace simultaneously
        ti = 0  # trace index
        for line in plan:
            m = re.match(GOAL_PATTERN, line)
            if m:
                if ti == len(trace_list):
                    return trace_list
                else:
                    raise EarlyGoalException()
            m = re.match(MAIN_PATTERN, line)
            if m is None:
                raise UnrecognizedLineException(line)
            action_type, predicate = m.group(1), m.group(2)
            predicate = self._conform_predicate(predicate)
            if action_type == "sync":
                if predicate == trace_list[ti]:
                    ti += 1  # good
                else:
                    raise PlanAndTraceDoNotMatchException(ti)
            elif action_type == "del":
                if predicate == trace_list[ti]:
                    trace_list.pop(ti)  # good
                    # don't advance ti: the rest of the list shifted 1 to the left,
                    # so there is a new item at position [ti] already
                else:
                    raise PlanAndTraceDoNotMatchException(ti)
            elif action_type == "add":
                trace_list.insert(ti, predicate)  # good
                ti += 1  # we just inserted a predicate in this position
            else:
                raise UnrecognizedActionTypeException(action_type)
