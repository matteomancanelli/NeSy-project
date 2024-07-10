import re

class UnrecognizedActionTypeException(Exception):
    pass
class UnrecognizedLineException(Exception):
    pass
class PlanAndTraceDoNotMatchException(Exception):
    pass
class NoPlanMatchedTheTraceException(Exception):
    pass

MAIN_PATTERN = re.compile(r"\( (.+?) - (.+) - (.+?) \)", re.VERBOSE)
GOAL_PATTERN = re.compile(r"\( gotogoal - (.+?) \)", re.VERBOSE)

class Aligner:

    def __init__(self, plans_file):
        self.plans_file = plans_file
        self.reset_all_plans()
    
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

    # This is IN PLACE
    def align(self, trace_list, trace_num):
        plan = self[trace_num]
        _align_to_plan(trace_list, plan)

    # This is IN PLACE
    def align_BRUTEFORCE(self, trace_list, _=None):
        matched = set()
        for i, plan in enumerate(self):
            try:
                _align_to_plan(trace_list, plan)
                # self.all_plans.pop(i)
                matched.add(i)
            except PlanAndTraceDoNotMatchException:
                pass
        if len(matched) > 0:
            print("Got", len(matched), "matches:", matched)
        else:
            raise NoPlanMatchedTheTraceException()

def _align_to_plan(trace_list, plan):
    # parse plan and align trace simultaneously
    ti = 0  # trace index
    for line in plan:
        m = re.match(GOAL_PATTERN, line)
        if m:
            if ti == len(trace_list):
                return
            else:
                raise PlanAndTraceDoNotMatchException()
        m = re.match(MAIN_PATTERN, line)
        if m is None:
            raise UnrecognizedLineException(line)
        action_type, predicate = m.group(1), m.group(2)
        if action_type == "sync":
            if predicate == trace_list[ti]:
                ti += 1  # good
            else:
                raise PlanAndTraceDoNotMatchException()
        elif action_type == "del":
            if predicate == trace_list[ti]:
                trace_list.pop(ti)  # good
                # don't advance ti: the rest of the list shifted 1 to the left,
                # so there is a new item at position [ti] already
            else:
                raise PlanAndTraceDoNotMatchException()
        elif action_type == "add":
            trace_list.insert(ti, predicate)  # good
            ti += 1  # we just inserted a predicate in this position
        else:
            raise UnrecognizedActionTypeException(action_type)
