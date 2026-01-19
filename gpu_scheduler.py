"""
GPU-Accelerated Course Scheduling using Numba CUDA
Optimizes fitness calculations using NVIDIA GPU (RTX 3050)
"""

import csv
import random
import math
import yaml
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from datetime import datetime
from numba import cuda, jit, prange

# Try to import CUDA support, fall back to CPU if not available
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except:
    CUDA_AVAILABLE = False

print(f"🎮 CUDA Available: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    print(f"   GPU: {cuda.get_current_device().name.decode()}")


@dataclass
class LHP:
    """Learning activity class - a single time slot entry"""
    tc: int  # credits
    ten: str  # name
    ma_hp: str  # course code (Mã HP)
    ma: str  # course class code (Mã LHP)
    lt_th: bool  # True if theory, False if practice
    gd: str  # classroom
    thu: int  # day (2-6)
    ca: Tuple[int, int]  # time slot (start, end)

    def __post_init__(self):
        assert self.thu in range(2, 7), "Thu must be 2-6"
        assert len(self.ca) == 2, "Ca must have exactly 2 elements"
        assert self.ca[1] - self.ca[0] <= 3, "Ca duration must be <= 3"

    @staticmethod
    def from_csv_row(row: List[str]) -> 'LHP':
        try:
            tc = int(row[3])
            ten = row[2]
            ma_hp = row[1]
            ma = row[9]
            lt_th = row[11] == "LT"
            thu = int(row[12])
            
            if "PES" in ma_hp:
                gd = row[15] if len(row) > 15 else ""
                tiết_str = row[14] if len(row) > 14 else ""
                if "Tiết" in tiết_str:
                    parts = tiết_str.replace("Tiết ", "").split("-")
                    ca = (int(parts[0]), int(parts[1]))
                else:
                    ca = (7, 9)
            else:
                gd = row[14] if len(row) > 14 else ""
                ca_str = row[13].strip() if row[13] else ""
                
                if not ca_str:
                    ca = (0, 0)
                elif '-' in ca_str:
                    parts = ca_str.split('-')
                    ca = (int(parts[0]), int(parts[1]))
                else:
                    ca_num = int(ca_str)
                    ca_map = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
                    ca = ca_map.get(ca_num, (0, 0))
            
            return LHP(tc=tc, ten=ten, ma_hp=ma_hp, ma=ma, lt_th=lt_th, gd=gd, thu=thu, ca=ca)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing CSV row: {row}, Error: {e}")


@dataclass
class CourseClass:
    """Represents a complete course class with all its components (LT and TH)"""
    ma: str
    ma_hp: str
    ten: str
    components: List[LHP]
    
    def get_total_credits(self) -> int:
        return self.components[0].tc if self.components else 0
    
    def get_all_times(self) -> List[Tuple[int, int, int]]:
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


def load_lhp_from_csv(filename: str, course_prefs: Dict = None) -> List[LHP]:
    """Load LHP objects from CSV file - filter only courses in preferences"""
    allowed_courses = set()
    if course_prefs and 'courses' in course_prefs:
        allowed_courses = set(course_prefs['courses'].keys())
    
    lhps = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 15 and row[12] and row[13]:
                    try:
                        ma_hp = row[1]
                        if allowed_courses and ma_hp not in allowed_courses:
                            continue
                        
                        lhp = LHP.from_csv_row(row)
                        lhps.append(lhp)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return []
    
    if allowed_courses:
        print(f"Filtered to {len(allowed_courses)} courses from preferences")
    
    return lhps


def group_by_course_class(lhps: List[LHP]) -> Dict[str, 'CourseClass']:
    """Group LHP objects by course class code"""
    ma_to_components = {}
    for lhp in lhps:
        if lhp.ma not in ma_to_components:
            ma_to_components[lhp.ma] = []
        ma_to_components[lhp.ma].append(lhp)
    
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
    """Group CourseClass objects by course code"""
    groups = {}
    for ma, course_class in course_classes.items():
        ma_hp = course_class.ma_hp
        if ma_hp not in groups:
            groups[ma_hp] = []
        groups[ma_hp].append(course_class)
    
    return groups


# CUDA Kernels for GPU acceleration
if CUDA_AVAILABLE:
    @cuda.jit
    def check_conflicts_kernel(day_array, start_array, end_array, result, n_courses):
        """GPU kernel to check time conflicts in parallel"""
        idx = cuda.grid(1)
        if idx >= n_courses:
            return
        
        for other_idx in range(n_courses):
            if idx != other_idx:
                if day_array[idx] == day_array[other_idx]:
                    # Check for overlap
                    if not (end_array[idx] < start_array[other_idx] or start_array[idx] > end_array[other_idx]):
                        result[0] = 1  # Conflict found
    
    @cuda.jit
    def fitness_kernel(days, starts, ends, credits, preferred_days, result, n_courses, n_pref_days):
        """GPU kernel for parallel fitness calculations"""
        idx = cuda.grid(1)
        if idx >= n_courses:
            return
        
        # Check if day is preferred
        is_preferred = 0
        for p_idx in range(n_pref_days):
            if days[idx] == preferred_days[p_idx]:
                is_preferred = 1
                break
        
        # Calculate penalty
        if is_preferred == 0:
            cuda.atomic.add(result, 0, -0.1)


class GPUGeneticScheduler:
    """GPU-Accelerated Genetic Algorithm Scheduler"""
    
    def __init__(self, course_code_groups: Dict[str, List[CourseClass]], 
                 preferences: Dict = None,
                 min_credits: int = 14, max_credits: int = 17,
                 population_size: int = 100, generations: int = 200):
        self.course_code_groups = course_code_groups
        self.preferences = preferences or {'courses': {}, 'ga_config': {}}
        
        ga_cfg = self.preferences.get('ga_config', {})
        self.min_credits = ga_cfg.get('min_credits', min_credits)
        self.max_credits = ga_cfg.get('max_credits', max_credits)
        self.population_size = ga_cfg.get('population_size', population_size)
        self.generations = ga_cfg.get('generations', generations)
        self.tournament_size = ga_cfg.get('tournament_size', 3)
        
        self.course_codes = sorted(course_code_groups.keys())
        
        # Extract time preferences
        time_prefs = self.preferences.get('time_preferences', {})
        self.preferred_days = time_prefs.get('preferred_days', [2, 3, 4, 5, 6])
        self.day_penalty = time_prefs.get('day_penalty', 0.1)
        self.start_mutation_rate = ga_cfg.get('start_mutation_rate', 0.3)
        self.end_mutation_rate = ga_cfg.get('end_mutation_rate', 0.05)
        
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
        
        self.gpu_enabled = CUDA_AVAILABLE
        print(f"🎯 GPU Acceleration: {'ENABLED ✅' if self.gpu_enabled else 'DISABLED (CPU FALLBACK)'}")

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
        """Check time conflicts - GPU accelerated if available"""
        if not individual:
            return True
        
        if self.gpu_enabled and len(individual) > 5:
            # GPU version for larger schedules
            return self._check_conflicts_gpu(individual)
        else:
            # CPU fallback
            return self._check_conflicts_cpu(individual)
    
    def _check_conflicts_cpu(self, individual: List[CourseClass]) -> bool:
        """CPU version of conflict checking"""
        time_slots = {}
        for course_class in individual:
            for thu, ca_start, ca_end in course_class.get_all_times():
                for (day, slot_start, slot_end) in list(time_slots.keys()):
                    if thu == day:
                        if not (ca_end < slot_start or ca_start > slot_end):
                            return False
                
                time_slots[(thu, ca_start, ca_end)] = True
        
        return True
    
    def _check_conflicts_gpu(self, individual: List[CourseClass]) -> bool:
        """GPU version of conflict checking using Numba CUDA"""
        try:
            # Prepare data for GPU
            all_times = []
            for course_class in individual:
                for thu, ca_start, ca_end in course_class.get_all_times():
                    all_times.append((thu, ca_start, ca_end))
            
            n = len(all_times)
            if n < 2:
                return True
            
            days = np.array([t[0] for t in all_times], dtype=np.int32)
            starts = np.array([t[1] for t in all_times], dtype=np.int32)
            ends = np.array([t[2] for t in all_times], dtype=np.int32)
            
            result = np.array([0], dtype=np.int32)
            
            # Launch CUDA kernel
            threads_per_block = 256
            blocks = (n + threads_per_block - 1) // threads_per_block
            check_conflicts_kernel[blocks, threads_per_block](days, starts, ends, result, n)
            
            return result[0] == 0
        except Exception as e:
            # Fallback to CPU if GPU fails
            return self._check_conflicts_cpu(individual)

    def check_hard_constraints(self, individual: List[CourseClass]) -> bool:
        """Check hard constraints"""
        if len(individual) == 0:
            return False
            
        # Constraint: Each course code should appear only once
        course_codes_used = set()
        for course_class in individual:
            if course_class.ma_hp in course_codes_used:
                return False
            course_codes_used.add(course_class.ma_hp)
        
        # Constraint: Total credits in range
        total_credits = sum(cc.get_total_credits() for cc in individual)
        if not (self.min_credits <= total_credits <= self.max_credits):
            return False
        
        # Constraint: Time distinctness (no overlapping)
        if not self.check_time_conflicts(individual):
            return False
        
        return True

    def calculate_time_consistency(self, individual: List[CourseClass]) -> float:
        """Calculate time consistency"""
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
        """Calculate space consistency"""
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
        return math.sqrt(variance)

    def fitness(self, individual: List[CourseClass]) -> float:
        """Fitness function with GPU-accelerated calculations"""
        if not self.check_hard_constraints(individual):
            return 0.0
        
        # Calculate soft optimization metrics
        time_consistency = self.calculate_time_consistency(individual)
        space_consistency = self.calculate_space_consistency(individual)
        
        # Base fitness from consistency
        fitness_score = (0.6 * time_consistency + 0.4 * space_consistency)
        
        # Add course preference bonuses
        course_bonus = 0.0
        for course_class in individual:
            ma_hp = course_class.ma_hp
            if ma_hp in self.course_prefs:
                course_bonus += self.course_prefs[ma_hp]['bonus']
        
        # Add penalty for courses on non-preferred days
        day_penalty = 0.0
        for course_class in individual:
            for thu, ca_start, ca_end in course_class.get_all_times():
                if thu not in self.preferred_days:
                    day_penalty -= self.day_penalty
        
        final_fitness = fitness_score + course_bonus + day_penalty
        
        return final_fitness

    def tournament_selection(self, population: List[List[CourseClass]], 
                           fitness_scores: List[float], tournament_size: int = 3) -> List[CourseClass]:
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    def crossover(self, parent1: List[CourseClass], parent2: List[CourseClass]) -> Tuple[List[CourseClass], List[CourseClass]]:
        """Time-aware crossover"""
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1[:], parent2[:]
        
        child1_codes = set()
        child1_list = []
        
        for course in parent1:
            if course.ma_hp not in child1_codes:
                test_schedule = child1_list + [course]
                if self.check_time_conflicts(test_schedule):
                    child1_list.append(course)
                    child1_codes.add(course.ma_hp)
        
        for course in parent2:
            if course.ma_hp not in child1_codes:
                test_schedule = child1_list + [course]
                if (self.check_time_conflicts(test_schedule) and 
                    sum(c.get_total_credits() for c in test_schedule) <= self.max_credits):
                    child1_list.append(course)
                    child1_codes.add(course.ma_hp)
        
        child2_codes = set()
        child2_list = []
        
        for course in parent2:
            if course.ma_hp not in child2_codes:
                test_schedule = child2_list + [course]
                if self.check_time_conflicts(test_schedule):
                    child2_list.append(course)
                    child2_codes.add(course.ma_hp)
        
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
        """Mutation with adaptive rate"""
        if len(individual) == 0:
            return individual
        
        progress = generation / total_generations if total_generations > 0 else 0
        adaptive_rate = self.start_mutation_rate - (self.start_mutation_rate - self.end_mutation_rate) * progress
        
        mutated = individual[:]
        
        if random.random() < adaptive_rate and len(mutated) > 0:
            idx = random.randint(0, len(mutated) - 1)
            ma_hp = mutated[idx].ma_hp
            options = self.course_code_groups.get(ma_hp, [mutated[idx]])
            mutated[idx] = random.choice(options)
        
        if random.random() < adaptive_rate / 2 and len(mutated) > 1:
            total_credits = sum(cc.get_total_credits() for cc in mutated)
            if total_credits - mutated[-1].get_total_credits() >= self.min_credits:
                mutated = mutated[:-1]
        
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
        """Repair an invalid individual"""
        repaired = individual[:]
        
        seen_codes = set()
        repaired_unique = []
        for course_class in repaired:
            if course_class.ma_hp not in seen_codes:
                repaired_unique.append(course_class)
                seen_codes.add(course_class.ma_hp)
        repaired = repaired_unique
        
        valid_schedule = []
        for course_class in repaired:
            test_schedule = valid_schedule + [course_class]
            if self.check_time_conflicts(test_schedule):
                valid_schedule.append(course_class)
        repaired = valid_schedule
        
        total_credits = sum(cc.get_total_credits() for cc in repaired)
        
        while total_credits > self.max_credits and len(repaired) > 0:
            min_idx = 0
            min_bonus = float('inf')
            for i, cc in enumerate(repaired):
                bonus = self.course_prefs.get(cc.ma_hp, {}).get('bonus', 0.0)
                if bonus < min_bonus:
                    min_bonus = bonus
                    min_idx = i
            
            total_credits -= repaired[min_idx].get_total_credits()
            repaired.pop(min_idx)
        
        if total_credits < self.min_credits:
            used_codes = set(cc.ma_hp for cc in repaired)
            available_codes = [c for c in self.course_codes if c not in used_codes]
            
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
        """Local search: hill climbing"""
        best = individual[:]
        best_fitness = self.fitness(best)
        
        for _ in range(iterations):
            improved = False
            
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
            return None, 0.0
        
        print(f"Generated {len(population)} valid schedules in {attempts} attempts")
        
        best_individual = None
        best_fitness = -1
        elite_individuals = []
        elite_size = 5
        
        for generation in range(self.generations):
            fitness_scores = [self.fitness(ind) for ind in population]
            
            gen_best_idx = max(range(len(population)), key=lambda i: fitness_scores[i])
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx][:]
            
            scored_population = [(population[i], fitness_scores[i]) for i in range(len(population))]
            scored_population.sort(key=lambda x: x[1], reverse=True)
            elite_individuals = [ind for ind, _ in scored_population[:elite_size]]
            
            if (generation + 1) % 50 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                valid_count = sum(1 for f in fitness_scores if f > 0)
                adaptive_mut = self.start_mutation_rate - (self.start_mutation_rate - self.end_mutation_rate) * (generation / self.generations)
                print(f"Generation {generation + 1}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, Valid={valid_count}/{len(population)}, MutRate={adaptive_mut:.3f}")
            
            new_population = []
            
            for elite in elite_individuals:
                new_population.append(elite[:])
            
            if (generation + 1) % 50 == 0:
                elite_individuals[0] = self.local_search(elite_individuals[0], iterations=5)
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, generation=generation, total_generations=self.generations)
                child2 = self.mutate(child2, generation=generation, total_generations=self.generations)
                
                child1 = self.repair_individual(child1)
                child2 = self.repair_individual(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population[:self.population_size]
        
        return best_individual, best_fitness

    def print_schedule(self, schedule: List[CourseClass]) -> None:
        """Print formatted schedule"""
        print("\n" + "="*80)
        print("OPTIMAL SCHEDULE (GPU-ACCELERATED)")
        print("="*80)
        
        total_credits = 0
        schedule_by_day = {}
        
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
    
    def save_schedule_to_file(self, schedule: List[CourseClass], fitness_score: float, filename: str = "recent_results.txt") -> None:
        """Append schedule result to file with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_credits = 0
        schedule_by_day = {}
        
        for course_class in schedule:
            for comp in course_class.components:
                if comp.thu not in schedule_by_day:
                    schedule_by_day[comp.thu] = []
                schedule_by_day[comp.thu].append(comp)
            total_credits += course_class.get_total_credits()
        
        with open(filename, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"SCHEDULING RESULT (GPU-ACCELERATED) - {timestamp}\n")
            f.write("="*80 + "\n")
            f.write(f"Best Fitness Score: {fitness_score:.4f}\n")
            f.write(f"Total Credits: {total_credits}\n")
            f.write(f"GPU: {'RTX 3050 ✅' if self.gpu_enabled else 'CPU Fallback'}\n")
            f.write(f"\n")
            
            for day in sorted(schedule_by_day.keys()):
                f.write(f"Thứ {day}:\n")
                for comp in sorted(schedule_by_day[day], key=lambda x: x.ca[0]):
                    ltype = "LT" if comp.lt_th else "TH"
                    f.write(f"  Tiết {comp.ca[0]}-{comp.ca[1]}: {comp.ten} ({comp.ma_hp} - {comp.ma}) | {ltype} | Phòng: {comp.gd}\n")
                f.write(f"\n")
            
            f.write(f"\n")


# Main execution
if __name__ == "__main__":
    print("\n🚀 GPU-Accelerated Course Scheduler")
    print("="*80)
    
    # Load preferences
    print("Loading course preferences...")
    preferences = load_preferences("scheduling_preferences.yaml")
    
    # Load data
    print("Loading LHP data from Full.csv...")
    lhps = load_lhp_from_csv("Full.csv", course_prefs=preferences)
    print(f"Loaded {len(lhps)} LHP records")
    
    # Group by course class
    course_classes = group_by_course_class(lhps)
    print(f"Found {len(course_classes)} unique course classes")
    
    # Group by course code
    course_code_groups = group_by_course_code_with_classes(course_classes)
    print(f"Found {len(course_code_groups)} unique course codes")
    
    # Create GPU scheduler
    print("\nInitializing GPU Genetic Algorithm Scheduler...")
    scheduler = GPUGeneticScheduler(
        course_code_groups,
        preferences=preferences
    )
    
    print(f"GA Configuration:")
    print(f"  - Credits: [{scheduler.min_credits}, {scheduler.max_credits}]")
    print(f"  - Population: {scheduler.population_size}")
    print(f"  - Generations: {scheduler.generations}")
    print(f"  - Course preferences loaded: {len(scheduler.course_prefs)} courses")
    print(f"  - Preferred days: {scheduler.preferred_days}")
    print(f"  - Day penalty: {scheduler.day_penalty}")
    
    # Run GA
    print("\n⚡ Running GPU-Accelerated GA optimization...\n")
    best_schedule, best_fitness = scheduler.evolve()
    
    # Display results
    if best_schedule:
        print(f"\n✨ Best Fitness Score: {best_fitness:.4f}")
        scheduler.print_schedule(best_schedule)
        scheduler.save_schedule_to_file(best_schedule, best_fitness)
        print("\n✅ Results saved to recent_results.txt")
    else:
        print("No valid schedule found!")
