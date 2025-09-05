#!/bin/bash

# Epic Decompose - Break epic into concrete, actionable tasks
# Usage: ./epic-decompose.sh <epic_name>

set -e

EPIC_NAME="$1"

if [ -z "$EPIC_NAME" ]; then
    echo "Usage: $0 <epic_name>"
    exit 1
fi

EPIC_DIR=".claude/epics/$EPIC_NAME"
EPIC_FILE="$EPIC_DIR/epic.md"

# Preflight checks
echo "🔍 Running preflight checks for epic: $EPIC_NAME"

# 1. Verify epic exists
if [ ! -f "$EPIC_FILE" ]; then
    echo "❌ Epic not found: $EPIC_NAME. First create it with: /pm:prd-parse $EPIC_NAME"
    exit 1
fi

# 2. Check for existing tasks
TASK_COUNT=0
EXISTING_TASKS=()
for task_file in "$EPIC_DIR"/*.md; do
    if [[ -f "$task_file" ]]; then
        filename=$(basename "$task_file")
        if [[ "$filename" =~ ^[0-9]+.*\.md$ ]] && [[ "$filename" != "epic.md" ]]; then
            TASK_COUNT=$((TASK_COUNT + 1))
            EXISTING_TASKS+=("$filename")
        fi
    fi
done

if [ $TASK_COUNT -gt 0 ]; then
    echo "⚠️ Found $TASK_COUNT existing tasks:"
    for task in "${EXISTING_TASKS[@]}"; do
        echo "  - $task"
    done
    read -p "Delete and recreate all tasks? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "View existing tasks with: /pm:epic-show $EPIC_NAME"
        exit 0
    fi
    
    # Remove existing task files
    for task_file in "${EXISTING_TASKS[@]}"; do
        rm -f "$EPIC_DIR/$task_file"
    done
    echo "🗑️ Removed $TASK_COUNT existing task files"
fi

# 3. Validate epic frontmatter
if ! grep -q "^name:" "$EPIC_FILE" || ! grep -q "^status:" "$EPIC_FILE" || ! grep -q "^created:" "$EPIC_FILE"; then
    echo "❌ Invalid epic frontmatter. Please check: $EPIC_FILE"
    exit 1
fi

# 4. Check epic status
EPIC_STATUS=$(grep "^status:" "$EPIC_FILE" | cut -d: -f2 | tr -d ' ' | head -1)
if [[ "$EPIC_STATUS" == "completed" ]]; then
    read -p "⚠️ Epic is marked as completed. Are you sure you want to decompose it again? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        exit 0
    fi
fi

echo "✅ Preflight checks passed"
echo

# Read the epic content
echo "📖 Analyzing epic: $EPIC_NAME"
EPIC_CONTENT=$(cat "$EPIC_FILE")

# Get current datetime for task creation
CURRENT_DATETIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Analyze epic content to determine task creation strategy
EPIC_SIZE="medium"  # Default assumption

# Count lines and complexity to determine size
LINE_COUNT=$(wc -l < "$EPIC_FILE")
if [ $LINE_COUNT -lt 100 ]; then
    EPIC_SIZE="small"
elif [ $LINE_COUNT -gt 300 ]; then
    EPIC_SIZE="large"
fi

echo "📊 Epic analysis: $EPIC_SIZE epic (${LINE_COUNT} lines)"

# For now, create tasks sequentially (can be enhanced later for parallel creation)
echo "🔨 Creating tasks for epic: $EPIC_NAME"

# Extract epic name for task titles
EPIC_TITLE=$(grep "^name:" "$EPIC_FILE" | cut -d: -f2- | sed 's/^ *//')

# Define common task structure based on epic content
# This is a template approach - in a full implementation, this would be more sophisticated
create_task() {
    local task_number="$1"
    local task_title="$2"
    local task_description="$3"
    local parallel="$4"
    local depends_on="$5"
    
    local task_file="$EPIC_DIR/$(printf "%03d" $task_number).md"
    
    cat > "$task_file" << EOF
---
name: $task_title
status: open
created: $CURRENT_DATETIME
updated: $CURRENT_DATETIME
github: [Will be updated when synced to GitHub]
depends_on: $depends_on
parallel: $parallel
conflicts_with: []
---

# Task: $task_title

## Description
$task_description

## Acceptance Criteria
- [ ] Implementation complete
- [ ] Tests written and passing
- [ ] Documentation updated

## Technical Details
- Implementation approach to be determined
- Code locations to be identified
- Key considerations to be evaluated

## Dependencies
- [ ] Review epic requirements
- [ ] Identify technical dependencies

## Effort Estimate
- Size: M
- Hours: 8-16
- Parallel: $parallel

## Definition of Done
- [ ] Code implemented
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Changes validated
EOF

    echo "  ✅ Created: $(printf "%03d" $task_number).md - $task_title"
}

# Create basic task structure based on epic analysis
# This is a simplified approach - ideally would parse epic content more intelligently

TASK_COUNTER=1

# Always create an analysis task first
create_task $TASK_COUNTER "Epic Analysis and Planning" \
    "Analyze the epic requirements and create detailed implementation plan" \
    "true" "[]"
TASK_COUNTER=$((TASK_COUNTER + 1))

# Look for common patterns in epic to suggest additional tasks
if grep -qi "test" "$EPIC_FILE"; then
    create_task $TASK_COUNTER "Test Infrastructure Setup" \
        "Set up testing framework and infrastructure for epic requirements" \
        "true" "[]"
    TASK_COUNTER=$((TASK_COUNTER + 1))
fi

if grep -qi "integration\|api" "$EPIC_FILE"; then
    create_task $TASK_COUNTER "Integration Implementation" \
        "Implement integration components and API endpoints" \
        "false" "[001]"
    TASK_COUNTER=$((TASK_COUNTER + 1))
fi

if grep -qi "documentation\|docs" "$EPIC_FILE"; then
    create_task $TASK_COUNTER "Documentation Update" \
        "Update documentation to reflect epic changes" \
        "true" "[]"
    TASK_COUNTER=$((TASK_COUNTER + 1))
fi

# Always create a validation/completion task
create_task $TASK_COUNTER "Epic Validation and Completion" \
    "Validate all epic requirements are met and complete final testing" \
    "false" "[$(seq -s, 1 $((TASK_COUNTER-1)))]"

TOTAL_TASKS=$TASK_COUNTER
PARALLEL_TASKS=$(grep -r "parallel: true" "$EPIC_DIR"/*.md 2>/dev/null | wc -l | tr -d ' ')
SEQUENTIAL_TASKS=$((TOTAL_TASKS - PARALLEL_TASKS))

echo
echo "✅ Created $TOTAL_TASKS tasks for epic: $EPIC_NAME"

# Update epic with task summary
TASK_SUMMARY=""
for i in $(seq 1 $TOTAL_TASKS); do
    task_file="$EPIC_DIR/$(printf "%03d" $i).md"
    if [ -f "$task_file" ]; then
        task_name=$(grep "^name:" "$task_file" | cut -d: -f2- | sed 's/^ *//')
        is_parallel=$(grep "^parallel:" "$task_file" | cut -d: -f2 | tr -d ' ')
        TASK_SUMMARY="$TASK_SUMMARY- [ ] $(printf "%03d" $i).md - $task_name (parallel: $is_parallel)\n"
    fi
done

# Add task summary to epic if not already present
if ! grep -q "## Tasks Created" "$EPIC_FILE"; then
    cat >> "$EPIC_FILE" << EOF

## Tasks Created
$TASK_SUMMARY
Total tasks: $TOTAL_TASKS
Parallel tasks: $PARALLEL_TASKS
Sequential tasks: $SEQUENTIAL_TASKS
Estimated total effort: $((TOTAL_TASKS * 12)) hours
EOF
fi

echo
echo "📊 Task Summary:"
echo "   Total tasks: $TOTAL_TASKS"
echo "   Parallel tasks: $PARALLEL_TASKS"  
echo "   Sequential tasks: $SEQUENTIAL_TASKS"
echo "   Estimated effort: $((TOTAL_TASKS * 12)) hours"
echo
echo "🚀 Ready to sync to GitHub? Run: /pm:epic-sync $EPIC_NAME"