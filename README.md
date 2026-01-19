# Course Scheduling Algorithm - Genetic Algorithm Optimizer

A sophisticated **Genetic Algorithm (GA) based course scheduler** that optimizes student schedules based on constraints, preferences, and consistency metrics. (VNU-UET specific)

## Overview

This project provides an intelligent scheduling system that:
- **Loads course data** from a CSV file (Full.csv)
- **Filters courses** based on user preferences (scheduling_preferences.yaml)
- **Optimizes schedules** using a multi-objective genetic algorithm
- **Respects hard constraints**: distinct courses, valid credit range, no time conflicts
- **Optimizes soft metrics**: time consistency, space consistency, course preferences
- **Produces valid schedules** with 99-100% population validity

## Features

### Hard Constraints (Must be satisfied)
1. **Distinct Courses**: No course code can appear twice in the same schedule
2. **Credit Range**: Total credits must be between 14-17 (configurable)
3. **No Time Conflicts**: No overlapping classes on the same day and time

### Soft Optimization (Maximized during evolution)
- **Time Consistency**: Minimize schedule fragmentation (courses grouped by day/time)
- **Space Consistency**: Optimize classroom distribution
- **Course Preferences**: Apply bonuses/penalties for preferred/avoided courses

### Advanced GA Features
- **Adaptive Mutation**: High exploration early (0.30), low exploitation late (0.05)
- **Repair Mechanism**: Automatically fixes invalid offspring instead of rejecting them
- **Local Search**: Hill-climbing on best solutions every 50 generations
- **Multi-Parent Elitism**: Preserves top 5 best individuals each generation
- **Time-Aware Crossover**: Intelligent recombination respecting time slot constraints

## Installation

### Prerequisites
- Python 3.7+
- Required packages: `pyyaml`

### Setup
```bash
# Install dependencies
pip install pyyaml

# Copy files to your working directory
# - scheduling.py (main script)
# - Full.csv (course data)
# - scheduling_preferences.yaml (configuration)
```

## Usage

### Quick Start
```bash
python scheduling.py
```

The script will:
1. Load preferences from `scheduling_preferences.yaml`
2. Load and filter course data from `Full.csv`
3. Initialize the genetic algorithm
4. Run 500 generations with population size 300
5. Display the optimal schedule

### Output Example
```
Loading course preferences...
Loaded preferences from scheduling_preferences.yaml
Loading LHP data from Full.csv...
Filtered to 7 courses from preferences
Loaded 174 LHP records
Found 74 unique course classes
Found 7 unique course codes

Initializing Genetic Algorithm Scheduler...
GA Configuration:
  - Credits: [14, 17]
  - Population: 300
  - Generations: 500
  - Course preferences loaded: 7 courses

Running GA optimization (this may take a while)...

Generating initial population...
Generated 300 valid schedules in 396 attempts
Generation 50: Best=0.9832, Avg=0.9638, Valid=295/300, MutRate=0.275
...
Generation 500: Best=0.9832, Avg=0.9822, Valid=300/300, MutRate=0.050

Best Fitness Score: 0.9832

============================================================================
OPTIMAL SCHEDULE
============================================================================

Thứ 2:
  Tiết 7-8: Giáo dục thể chất cơ bản (PES 1003 - PES 1003 1) | TH | Phòng: TT GDTC
Thứ 3:
  Tiết 9-11: Giải tích 2 (UET.MAT1051 - UET.MAT1051 45) | TH | Phòng: 307-B
...
Tổng tín chỉ: 14
```

## Configuration

### scheduling_preferences.yaml

This YAML file controls which courses to consider and how to optimize them.

#### Structure

```yaml
courses:
  "COURSE_CODE":
    weight: 0.0-1.0      # Preference weight (0=avoid, 1=must-have)
    bonus: 0.0-0.5       # Fitness bonus added to final score
    description: "..."   # Human-readable description

time_preferences:
  preferred_days: [2, 3, 4, 5, 6]    # Monday-Friday (2-6)
  avoid_early_morning: false
  avoid_late_evening: false
  preferred_session_start: 4

ga_config:
  min_credits: 14        # Minimum credits required
  max_credits: 17        # Maximum credits allowed
  population_size: 300   # GA population size
  generations: 500       # Number of generations to evolve
  tournament_size: 3     # Tournament selection size
```

#### Example Configuration

```yaml
courses:
  "INT3403":
    weight: 1.0
    bonus: 0.5
    description: "Đồ họa máy tính - Computer Graphics"
  
  "UET.MAT1051":
    weight: 0.7
    bonus: 0.15
    description: "Giải tích 2 - Calculus 2"
  
  "PES 1003":
    weight: 0.6
    bonus: 0.05
    description: "Giáo dục thể chất - Physical Education"

ga_config:
  min_credits: 14
  max_credits: 17
  population_size: 300
  generations: 500
  tournament_size: 3
```

### Customizing Preferences

#### Add a Course
```yaml
courses:
  "INT3418":              # Must match Mã HP in CSV exactly
    weight: 0.8
    bonus: 0.2
    description: "Advanced Algorithms"
```

#### Change Credit Requirements
```yaml
ga_config:
  min_credits: 12         # Lower minimum
  max_credits: 20         # Higher maximum
```

#### Increase Algorithm Intensity
```yaml
ga_config:
  population_size: 500    # Larger population
  generations: 1000       # More generations
```

#### Add a "Must-Have" Course
```yaml
courses:
  "REQUIRED_COURSE":
    weight: 1.0           # Highest priority
    bonus: 0.5            # Highest bonus
    description: "..."
```

## CSV Data Format

The `Full.csv` file should have columns:
- **Col 0**: Lớp (Class)
- **Col 1**: Mã HP (Course Code) - **FILTERED BY YAML**
- **Col 2**: Môn (Course Name)
- **Col 3**: TC (Credits)
- **Col 9**: Mã LHP (Class Code)
- **Col 11**: LT/TH (Lecture/Practice)
- **Col 12**: Thứ (Day: 2-6 for Mon-Fri)
- **Col 13**: Ca (Time Slot - number for normal courses, text for PES)
- **Col 14**: GĐ (Classroom)

### Special Handling
- **Normal Courses**: `Ca` column contains a number (1-4) converted to time slots
- **PES Courses**: `Ca` column ignored; classroom from col 15; time format "Tiết X-Y" from col 14

## Algorithm Details

### Genetic Algorithm Components

1. **Individual Representation**: List of selected CourseClass objects
2. **Population**: 300 individuals (configurable)
3. **Generations**: 500 iterations (configurable)

### Selection
- **Tournament Selection**: Randomly select 3 individuals, pick the best

### Crossover (Time-Aware)
- Intelligently combines courses from two parents
- Respects time slot constraints
- Prefers parent1's courses first, fills gaps from parent2

### Mutation (Adaptive)
- **Replace**: Swap course with alternative class (30% early → 5% late)
- **Remove**: Drop lowest-preference course if possible
- **Add**: Attempt to add new compatible course

### Repair Mechanism
- Removes duplicate courses
- Eliminates time conflicts
- Adjusts credits to valid range by removing/adding courses
- **Guarantees valid offspring** (99-100% population validity)

### Local Search
- Hill-climbing on best individual every 50 generations
- Tries 1-step variations of course selections
- Improves solution quality without constraint violation

### Elitism
- Preserves top 5 best individuals each generation
- Ensures best solution never degrades
- Applies local search to elite occasionally

## Fitness Function

```
Fitness = Base_Fitness + Course_Bonuses

Base_Fitness = 0.6 * Time_Consistency + 0.4 * Space_Consistency

Time_Consistency = 1 / (1 + day_variance + time_variance)
                 # High when courses grouped by day/time

Space_Consistency = ideal_rooms / actual_rooms
                  # High when using few, balanced classrooms

Course_Bonuses = SUM of bonus values for selected courses from YAML
               # INT3403 adds +0.5, UET.MAT1051 adds +0.15, etc.
```

## Performance Metrics

Typical results on standard 7-course schedule:
- **Fitness Score**: 0.93-0.98
- **Population Validity**: 99-100%
- **Runtime**: ~60-120 seconds (500 generations × 300 population)
- **Credit Accuracy**: ±0.5% from target

## Troubleshooting

### "Error: File not found"
- Verify `Full.csv` is in the same directory
- Check filename spelling matches exactly

### "Could not generate valid initial population"
- Constraints too tight (e.g., min_credits=20 with only 3 courses)
- Relaxed credit constraints in YAML
- Add more courses to preferences

### Poor Schedule Quality
- Increase `generations` in YAML (e.g., 1000+)
- Increase `population_size` (e.g., 500)
- Adjust course preferences (higher bonuses for preferred courses)

### "Warning: CSV row parse error"
- CSV format may have encoding issues
- Verify UTF-8 encoding
- Check column count matches expected format

## Advanced Usage

### Batch Run with Different Preferences
```bash
# Create multiple preference files
cp scheduling_preferences.yaml preferences_v1.yaml
# Edit preferences_v1.yaml...

# Modify main section to load preferred file:
preferences = load_preferences("preferences_v1.yaml")
```

### Export Results
Add to the `print_schedule` method to save to file:
```python
with open("schedule_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Best Fitness: {best_fitness}\n")
    # ... write schedule details
```

## Class Reference

### LHP
- **Attributes**: tc, ten, ma_hp, ma, lt_th, gd, thu, ca
- **Method**: `from_csv_row(row)` - Parse CSV row

### CourseClass
- **Attributes**: ma, ma_hp, ten, components
- **Methods**: 
  - `get_total_credits()` - Total credits for this class
  - `get_all_times()` - List of (day, start_time, end_time) tuples

### GeneticScheduler
- **Main Methods**:
  - `evolve()` - Run GA optimization
  - `fitness(individual)` - Calculate schedule fitness
  - `check_hard_constraints(individual)` - Verify hard constraints
  - `repair_individual(individual)` - Fix invalid schedules
  - `local_search(individual)` - Hill-climbing optimization
  - `print_schedule(schedule)` - Display formatted output

## License & Attribution

This scheduling algorithm uses:
- **Genetic Algorithm** with tournament selection, adaptive crossover/mutation
- **Constraint satisfaction** for hard constraints
- **Multi-objective optimization** combining consistency + preferences

## Future Enhancements

Potential improvements:
- [ ] Multi-objective Pareto front tracking
- [ ] ACO (Ant Colony Optimization) hybrid
- [ ] GUI interface for preference editing
- [ ] Export to iCal/Google Calendar format
- [ ] Machine learning for preference learning
- [ ] Real-time conflict detection
- [ ] Student group scheduling (optimize for multiple students)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify YAML syntax (no tabs, valid indentation)
3. Ensure CSV matches expected format
4. Review fitness score and population validity metrics

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Python Version**: 3.7+
