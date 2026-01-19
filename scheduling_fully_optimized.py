import csv
import random
import math
import yaml
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field


@dataclass
class LHP:
    """Learning activity class - a single time slot entry"""
    tc: int  # credits
    ten: str  # name
    ma_hp: str  # course code (Mã HP) - for grouping distinct courses
    ma: str  # course class code (Mã LHP) - for scheduling options
    lt_th: bool  # True if theory, False if practice
    gd: str  # classroom
    thu: int  # day (2-6)
    ca: Tuple[int, int]  # time slot (start, end)

    def __post_init__(self):
        assert self.thu in range(2, 7), "Thursday must be 2-6"
        assert len(self.ca) == 2, "Ca must have exactly 2 elements"
        assert self.ca[1] - self.ca[0] <= 3, "Ca duration must be <= 3"

    @staticmethod
    def from_csv_row(row: List[str]) -> 'LHP':
        """
        Parse CSV row and create LHP object
        CSV columns: 0=Lớp, 1=Mã HP, 2=Môn, 3=TC, ..., 9=Mã LHP, 11=LT/TH, 12=Thứ, 13=Ca, 14=GĐ
        """
        try:
            tc = int(row[3])
            ten = row[2]
            ma_hp = row[1]  # Course code (e.g., "UET.MAT1051")
            ma = row[9]  # Course class code (e.g., "UET.MAT1051 1")
            lt_th = row[11] == "LT"  # True if theory class
            gd = row[14]
            thu = int(row[12])
            
            # Parse time slot (ca)
            if "PES" in row[1]:
                ca_str = row[13]
                if "Tiết" in ca_str:
                    parts = ca_str.replace("Tiết ", "").split("-")
                    ca = (int(parts[0]), int(parts[1]))
                else:
                    ca = (int(row[13]), int(row[13]) + 1)
            else:
                ca_num = int(row[13])
                ca = (ca_num * 4 - 3, ca_num * 4 - 1)
            
            return LHP(tc=tc, ten=ten, ma_hp=ma_hp, ma=ma, lt_th=lt_th, gd=gd, thu=thu, ca=ca)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing CSV row: {row}, Error: {e}")


@dataclass
class CourseClass:
    """Represents a complete course class with all its components (LT and TH)"""
    ma: str  # course class code (e.g., "UET.MAT1051 50")
    ma_hp: str  # course code (e.g., "UET.MAT1051")
    ten: str  # course name
    components: List[LHP]  # all LT and TH components
    
    def get_total_credits(self) -> int:
        """Get total credits (should be same for all components of a class)"""
        return self.components[0].tc if self.components else 0
    
    def get_all_times(self) -> List[Tuple[int, int, int]]:
        """Get all time slots (thu, ca_start, ca_end) for all components"""
        times = []
        for comp in self.components:
            times.append((comp.thu, comp.ca[0], comp.ca[1]))
        return times


def load_preferences(filename: str = "scheduling_preferences.yaml") -> Dict:
    """Load course preferences from YAML file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            prefs = yaml.safe_load(f)
        print(f"Loaded preferences from {filename}")
        return prefs
    except FileNotFoundError:
        print(f"Warning: Preferences file {filename} not found. Using defaults.")
        return {
            'courses': {},
            'time_preferences': {},
            'ga_config': {
                'min_credits': 14,
                'max_credits': 17,
                'population_size': 300,
                'generations': 500,
                'tournament_size': 3
            }
        }


def load_lhp_from_csv(filename: str) -> List[LHP]:
    """Load all LHP objects from CSV file"""
    lhps = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 15 and row[12] and row[13]:  # ensure valid row
                    try:
                        lhp = LHP.from_csv_row(row)
                        lhps.append(lhp)
                    except ValueError as e:
                        print(f"Warning: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return []
    
    return lhps


def group_by_course_class(lhps: List[LHP]) -> Dict[str, CourseClass]:
    """
    Group LHP objects by 'ma' (course class code).
    Each course class includes all its components (LT and TH).
    Returns a dict of ma -> CourseClass
    """
    ma_to_components = {}
    for lhp in lhps:
        if lhp.ma not in ma_to_components:
            ma_to_components[lhp.ma] = []
        ma_to_components[lhp.ma].append(lhp)
    
    # Create CourseClass objects
    course_classes = {}
    for ma, components in ma_to_components.items():
        if components:
            course_classes[ma] = CourseClass(
                ma=ma,
                ma_hp=components[0].ma_hp,
                ten=components[0].ten,
                components=components
            )
    
    return course_classes


def group_by_course_code_with_classes(course_classes: Dict[str, CourseClass]) -> Dict[str, List[CourseClass]]:
    """
    Group CourseClass objects by 'ma_hp' (course code).
    Each course code has multiple class options.
    """
    groups = {}
    for ma, course_class in course_classes.items():
        ma_hp = course_class.ma_hp
        if ma_hp not in groups:
            groups[ma_hp] = []
        groups[ma_hp].append(course_class)
    
    return groups


class GeneticScheduler:
    """Genetic Algorithm-based scheduler for LHP courses"""
    
    def __init__(self, course_code_groups: Dict[str, List[CourseClass]], 
                 preferences: Dict = None,
                 min_credits: int = 14, max_credits: int = 17,
                 population_size: int = 100, generations: int = 200):
        self.course_code_groups = course_code_groups
        self.preferences = preferences or {'courses': {}, 'ga_config': {}}
        
        # Get values from preferences or use defaults
        ga_cfg = self.preferences.get('ga_config', {})
        self.min_credits = ga_cfg.get('min_credits', min_credits)
        self.max_credits = ga_cfg.get('max_credits', max_credits)
        self.population_size = ga_cfg.get('population_size', population_size)
        self.generations = ga_cfg.get('generations', generations)
        self.tournament_size = ga_cfg.get('tournament_size', 3)
        
        self.course_codes = sorted(course_code_groups.keys())
        
        # Extract course preferences
        self.course_prefs = {}
        for ma_hp, pref_data in self.preferences.get('courses', {}).items():
            if isinstance(pref_data, dict):
                self.course_prefs[ma_hp] = {
                    'weight': pref_data.get('weight', 0.5),
                    'bonus': pref_data.get('bonus', 0.0)
                }
            else:
                self.course_prefs[ma_hp] = {'weight': 0.5, 'bonus': 0.0}

    def create_individual(self) -> List[CourseClass]:
        """Create a random individual (schedule) with valid credit range"""
        schedule = []
        total_credits = 0
        used_course_codes = set()
        
        shuffled_codes = self.course_codes[:]
        random.shuffle(shuffled_codes)
        
        for ma_hp in shuffled_codes:
            if ma_hp in used_course_codes:
                continue
                
            if total_credits >= self.max_credits:
                break
            
            course_options = self.course_code_groups[ma_hp]
            selected = random.choice(course_options)
            
            test_schedule = schedule + [selected]
            if (self.check_time_conflicts(test_schedule) and 
                total_credits + selected.get_total_credits() <= self.max_credits):
                schedule.append(selected)
                used_course_codes.add(ma_hp)
                total_credits += selected.get_total_credits()
        
        return schedule

    def check_time_conflicts(self, individual: List[CourseClass]) -> bool:
        """Check if there are time conflicts (overlapping classes on same day/time)"""
        time_slots = {}
        for course_class in individual:
            for thu, ca_start, ca_end in course_class.get_all_times():
                # Check overlap with existing slots on the same day
                for (day, slot_start, slot_end) in list(time_slots.keys()):
                    if thu == day:
                        if not (ca_end < slot_start or ca_start > slot_end):
                            return False
                
                time_slots[(thu, ca_start, ca_end)] = True
        
        return True

    def check_hard_constraints(self, individual: List[CourseClass]) -> bool:
        """Check hard constraints: credit range, time distinctness, and course distinctness"""
        if len(individual) == 0:
            return False
            
        # Constraint 3: Each course code should appear only once
        course_codes_used = set()
        for course_class in individual:
            if course_class.ma_hp in course_codes_used:
                return False
            course_codes_used.add(course_class.ma_hp)
        
        # Constraint 2: Total credits in range [14, 17]
        total_credits = sum(cc.get_total_credits() for cc in individual)
        if not (self.min_credits <= total_credits <= self.max_credits):
            return False
        
        # Constraint 1: Time distinctness (no overlapping)
        if not self.check_time_conflicts(individual):
            return False
        
        return True

    def calculate_time_consistency(self, individual: List[CourseClass]) -> float:
        """Calculate time consistency (lower variance = higher score)"""
        if len(individual) < 2:
            return 1.0
        
        days = []
        times = []
        for course_class in individual:
            for thu, ca_start, ca_end in course_class.get_all_times():
                days.append(thu)
                times.append(ca_start)
                times.append(ca_end)
        
        day_variance = self._calculate_variance(days)
        time_variance = self._calculate_variance(times)
        
        consistency = 1.0 / (1.0 + day_variance + time_variance)
        return consistency

    def calculate_space_consistency(self, individual: List[CourseClass]) -> float:
        """Calculate space consistency (how many different classrooms are used)"""
        classrooms = set()
        for course_class in individual:
            for comp in course_class.components:
                classrooms.add(comp.gd)
        
        ideal_rooms = max(1, len(individual) // 3)
        consistency = ideal_rooms / len(classrooms) if classrooms else 0
        return min(consistency, 1.0)

    @staticmethod
    def _calculate_variance(values: List[int]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)  # Use standard deviation

    def fitness(self, individual: List[CourseClass]) -> float:
        """
        Fitness function combining hard constraints and soft optimization.
        Hard constraints: must satisfy to have non-zero fitness
        Soft optimization: consistency metrics + course preferences
        """
        # Check hard constraints
        if not self.check_hard_constraints(individual):
            return 0.0
        
        # Calculate soft optimization metrics
        time_consistency = self.calculate_time_consistency(individual)
        space_consistency = self.calculate_space_consistency(individual)
        
        # Base fitness from consistency
        fitness_score = (0.6 * time_consistency + 0.4 * space_consistency)
        
        # Add course preference bonuses from YAML config
        course_bonus = 0.0
        for course_class in individual:
            ma_hp = course_class.ma_hp
            if ma_hp in self.course_prefs:
                course_bonus += self.course_prefs[ma_hp]['bonus']
        
        # Combine base fitness with course bonus
        final_fitness = fitness_score + course_bonus
        
        return final_fitness

    def tournament_selection(self, population: List[List[CourseClass]], 
                           fitness_scores: List[float], tournament_size: int = 3) -> List[CourseClass]:
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    def crossover(self, parent1: List[CourseClass], parent2: List[CourseClass]) -> Tuple[List[CourseClass], List[CourseClass]]:
        """Time-aware crossover: intelligently combines courses respecting time slots"""
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1[:], parent2[:]
        
        # Extract courses by time compatibility
        parent1_by_time = {}  # day -> list of courses
        parent2_by_time = {}
        
        for course in parent1:
            for thu, ca_start, ca_end in course.get_all_times():
                time_key = (thu, ca_start, ca_end)
                if time_key not in parent1_by_time:
                    parent1_by_time[time_key] = []
                parent1_by_time[time_key].append(course.ma_hp)
        
        for course in parent2:
            for thu, ca_start, ca_end in course.get_all_times():
                time_key = (thu, ca_start, ca_end)
                if time_key not in parent2_by_time:
                    parent2_by_time[time_key] = []
                parent2_by_time[time_key].append(course.ma_hp)
        
        # Build child1: prefer parent1's courses, fill gaps from parent2
        child1_codes = set()
        child1_list = []
        
        # First pass: take courses from parent1
        for course in parent1:
            if course.ma_hp not in child1_codes:
                test_schedule = child1_list + [course]
                if self.check_time_conflicts(test_schedule):
                    child1_list.append(course)
                    child1_codes.add(course.ma_hp)
        
        # Second pass: fill from parent2 if time permits
        for course in parent2:
            if course.ma_hp not in child1_codes:
                test_schedule = child1_list + [course]
                if (self.check_time_conflicts(test_schedule) and 
                    sum(c.get_total_credits() for c in test_schedule) <= self.max_credits):
                    child1_list.append(course)
                    child1_codes.add(course.ma_hp)
        
        # Build child2: prefer parent2's courses, fill gaps from parent1
        child2_codes = set()
        child2_list = []
        
        # First pass: take courses from parent2
        for course in parent2:
            if course.ma_hp not in child2_codes:
                test_schedule = child2_list + [course]
                if self.check_time_conflicts(test_schedule):
                    child2_list.append(course)
                    child2_codes.add(course.ma_hp)
        
        # Second pass: fill from parent1 if time permits
        for course in parent1:
            if course.ma_hp not in child2_codes:
                test_schedule = child2_list + [course]
                if (self.check_time_conflicts(test_schedule) and 
                    sum(c.get_total_credits() for c in test_schedule) <= self.max_credits):
                    child2_list.append(course)
                    child2_codes.add(course.ma_hp)
        
        return child1_list, child2_list

    def mutate(self, individual: List[CourseClass], mutation_rate: float = 0.15, 
               generation: int = 0, total_generations: int = 500) -> List[CourseClass]:
        """Mutation: replace, add, or remove courses. Rate adapts over time (high early, low late)"""
        if len(individual) == 0:
            return individual
        
        # Adaptive mutation rate: start at 0.3, decrease to 0.05 over generations
        progress = generation / total_generations if total_generations > 0 else 0
        adaptive_rate = 0.3 - (0.3 - 0.05) * progress
        
        mutated = individual[:]
        
        # Mutation type 1: Replace existing course with alternative class
        if random.random() < adaptive_rate and len(mutated) > 0:
            idx = random.randint(0, len(mutated) - 1)
            ma_hp = mutated[idx].ma_hp
            options = self.course_code_groups.get(ma_hp, [mutated[idx]])
            mutated[idx] = random.choice(options)
        
        # Mutation type 2: Remove a course
        if random.random() < adaptive_rate / 2 and len(mutated) > 1:
            total_credits = sum(cc.get_total_credits() for cc in mutated)
            if total_credits - mutated[-1].get_total_credits() >= self.min_credits:
                mutated = mutated[:-1]
        
        # Mutation type 3: Try to add a new course
        if random.random() < adaptive_rate / 2:
            total_credits = sum(cc.get_total_credits() for cc in mutated)
            used_codes = set(cc.ma_hp for cc in mutated)
            available_codes = [code for code in self.course_codes if code not in used_codes]
            
            if available_codes and total_credits < self.max_credits:
                ma_hp = random.choice(available_codes)
                options = self.course_code_groups[ma_hp]
                new_course = random.choice(options)
                total_credits += new_course.get_total_credits()
                
                if total_credits <= self.max_credits:
                    test_schedule = mutated + [new_course]
                    if self.check_time_conflicts(test_schedule):
                        mutated.append(new_course)
        
        return mutated

    def repair_individual(self, individual: List[CourseClass]) -> List[CourseClass]:
        """Repair an invalid individual to make it valid"""
        repaired = individual[:]
        
        # Step 1: Remove duplicate courses (keep first occurrence of each ma_hp)
        seen_codes = set()
        repaired_unique = []
        for course_class in repaired:
            if course_class.ma_hp not in seen_codes:
                repaired_unique.append(course_class)
                seen_codes.add(course_class.ma_hp)
        repaired = repaired_unique
        
        # Step 2: Remove courses that cause time conflicts
        valid_schedule = []
        for course_class in repaired:
            test_schedule = valid_schedule + [course_class]
            if self.check_time_conflicts(test_schedule):
                valid_schedule.append(course_class)
        repaired = valid_schedule
        
        # Step 3: Adjust credit count to be within range
        total_credits = sum(cc.get_total_credits() for cc in repaired)
        
        # If too many credits, remove courses greedily (least preferred first)
        while total_credits > self.max_credits and len(repaired) > 0:
            # Remove course with lowest preference score
            min_idx = 0
            min_bonus = float('inf')
            for i, cc in enumerate(repaired):
                bonus = self.course_prefs.get(cc.ma_hp, {}).get('bonus', 0.0)
                if bonus < min_bonus:
                    min_bonus = bonus
                    min_idx = i
            
            total_credits -= repaired[min_idx].get_total_credits()
            repaired.pop(min_idx)
        
        # If too few credits and below minimum, try to add courses
        if total_credits < self.min_credits:
            used_codes = set(cc.ma_hp for cc in repaired)
            available_codes = [c for c in self.course_codes if c not in used_codes]
            
            # Sort available by preference (highest first)
            available_codes.sort(
                key=lambda c: self.course_prefs.get(c, {}).get('bonus', 0.0),
                reverse=True
            )
            
            for ma_hp in available_codes:
                if total_credits >= self.min_credits:
                    break
                
                options = self.course_code_groups[ma_hp]
                new_course = random.choice(options)
                
                if total_credits + new_course.get_total_credits() <= self.max_credits:
                    test_schedule = repaired + [new_course]
                    if self.check_time_conflicts(test_schedule):
                        repaired.append(new_course)
                        total_credits += new_course.get_total_credits()
        
        return repaired

    def local_search(self, individual: List[CourseClass], iterations: int = 10) -> List[CourseClass]:
        """Local search: hill climbing by trying 1-step swaps and variations"""
        best = individual[:]
        best_fitness = self.fitness(best)
        
        for _ in range(iterations):
            improved = False
            
            # Try replacing each course with alternatives
            for i in range(len(best)):
                ma_hp = best[i].ma_hp
                options = self.course_code_groups.get(ma_hp, [best[i]])
                
                for option in options:
                    if option == best[i]:
                        continue
                    
                    test_individual = best[:]
                    test_individual[i] = option
                    test_individual = self.repair_individual(test_individual)
                    test_fitness = self.fitness(test_individual)
                    
                    if test_fitness > best_fitness:
                        best = test_individual
                        best_fitness = test_fitness
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best

    def evolve(self) -> Tuple[List[CourseClass], float]:
        """Run GA to find optimal schedule"""
        # Initialize population with only valid individuals
        print("Generating initial population...")
        population = []
        attempts = 0
        max_attempts = self.population_size * 50
        
        while len(population) < self.population_size and attempts < max_attempts:
            individual = self.create_individual()
            if self.check_hard_constraints(individual):
                population.append(individual)
            attempts += 1
        
        if len(population) == 0:
            print("Error: Could not generate any valid initial population!")
            print(f"Attempted {attempts} times with constraints:")
            print(f"  - Credits: [{self.min_credits}, {self.max_credits}]")
            print(f"  - No time conflicts")
            return None, 0.0
        
        print(f"Generated {len(population)} valid schedules in {attempts} attempts")
        
        best_individual = None
        best_fitness = -1
        elite_individuals = []  # Keep top 3-5 best individuals
        elite_size = 5
        
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = [self.fitness(ind) for ind in population]
            
            # Track best individual
            gen_best_idx = max(range(len(population)), key=lambda i: fitness_scores[i])
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx][:]
            
            # Update elite population (keep top elite_size individuals)
            scored_population = [(population[i], fitness_scores[i]) for i in range(len(population))]
            scored_population.sort(key=lambda x: x[1], reverse=True)
            elite_individuals = [ind for ind, _ in scored_population[:elite_size]]
            
            # Print progress
            if (generation + 1) % 50 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                valid_count = sum(1 for f in fitness_scores if f > 0)
                adaptive_mut = 0.3 - (0.3 - 0.05) * (generation / self.generations)
                print(f"Generation {generation + 1}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, Valid={valid_count}/{len(population)}, MutRate={adaptive_mut:.3f}")
            
            # Create new population through selection, crossover, mutation
            new_population = []
            
            # Multi-parent elitism: keep top 5 individuals + apply local search to best
            for elite in elite_individuals:
                new_population.append(elite[:])
            
            # Apply local search to best elite occasionally
            if (generation + 1) % 50 == 0:  # Every 50 generations
                elite_individuals[0] = self.local_search(elite_individuals[0], iterations=5)
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, generation=generation, total_generations=self.generations)
                child2 = self.mutate(child2, generation=generation, total_generations=self.generations)
                
                # Apply repair mechanism to all offspring
                child1 = self.repair_individual(child1)
                child2 = self.repair_individual(child2)
                
                # Add repaired children (now guaranteed to be valid)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population[:self.population_size]
        
        return best_individual, best_fitness

    def print_schedule(self, schedule: List[CourseClass]) -> None:
        """Print formatted schedule"""
        print("\n" + "="*80)
        print("OPTIMAL SCHEDULE")
        print("="*80)
        
        total_credits = 0
        schedule_by_day = {}
        
        # Flatten all components by day
        for course_class in schedule:
            for comp in course_class.components:
                if comp.thu not in schedule_by_day:
                    schedule_by_day[comp.thu] = []
                schedule_by_day[comp.thu].append(comp)
            total_credits += course_class.get_total_credits()
        
        for day in sorted(schedule_by_day.keys()):
            print(f"\nThứ {day}:")
            for comp in sorted(schedule_by_day[day], key=lambda x: x.ca[0]):
                ltype = "LT" if comp.lt_th else "TH"
                print(f"  Tiết {comp.ca[0]}-{comp.ca[1]}: {comp.ten} ({comp.ma_hp} - {comp.ma}) | {ltype} | Phòng: {comp.gd}")
        
        print(f"\nTổng tín chỉ: {total_credits}")
        print("="*80)


# Main execution
if __name__ == "__main__":
    # Load preferences
    print("Loading course preferences...")
    preferences = load_preferences("scheduling_preferences.yaml")
    
    # Load data
    print("Loading LHP data from Available.csv...")
    lhps = load_lhp_from_csv("Available.csv")
    print(f"Loaded {len(lhps)} LHP records")
    
    # Group by course class (ma) - each class includes all its LT and TH components
    course_classes = group_by_course_class(lhps)
    print(f"Found {len(course_classes)} unique course classes")
    
    # Group by course code to get multiple class options per course
    course_code_groups = group_by_course_code_with_classes(course_classes)
    print(f"Found {len(course_code_groups)} unique course codes")
    
    # Create scheduler with preferences
    print("\nInitializing Genetic Algorithm Scheduler...")
    scheduler = GeneticScheduler(
        course_code_groups,
        preferences=preferences
    )
    
    print(f"GA Configuration:")
    print(f"  - Credits: [{scheduler.min_credits}, {scheduler.max_credits}]")
    print(f"  - Population: {scheduler.population_size}")
    print(f"  - Generations: {scheduler.generations}")
    print(f"  - Course preferences loaded: {len(scheduler.course_prefs)} courses")
    
    # Run GA
    print("\nRunning GA optimization (this may take a while)...\n")
    best_schedule, best_fitness = scheduler.evolve()
    
    # Display results
    if best_schedule:
        print(f"\nBest Fitness Score: {best_fitness:.4f}")
        scheduler.print_schedule(best_schedule)
    else:
        print("No valid schedule found!")
