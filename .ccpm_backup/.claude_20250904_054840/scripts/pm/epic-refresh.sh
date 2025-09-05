#!/bin/bash

# Epic Refresh - Update epic progress based on task states
# Usage: ./epic-refresh.sh <epic_name>

set -e

EPIC_NAME="$1"

if [ -z "$EPIC_NAME" ]; then
    echo "Usage: $0 <epic_name>"
    exit 1
fi

EPIC_DIR=".claude/epics/$EPIC_NAME"

if [ ! -d "$EPIC_DIR" ]; then
    echo "❌ Epic directory not found: $EPIC_DIR"
    exit 1
fi

echo "🔄 Refreshing epic: $EPIC_NAME"
echo

# Count task status
TOTAL_TASKS=0
CLOSED_TASKS=0
OPEN_TASKS=0
IN_PROGRESS_TASKS=0

# Find all task files (numbered .md files)
for task_file in "$EPIC_DIR"/*.md; do
    if [[ ! -f "$task_file" ]] || [[ "$(basename "$task_file")" == "epic.md" ]]; then
        continue
    fi
    
    # Skip if not numbered task file
    if [[ ! "$(basename "$task_file")" =~ ^[0-9] ]]; then
        continue
    fi
    
    TOTAL_TASKS=$((TOTAL_TASKS + 1))
    
    # Check status in frontmatter
    if grep -q "^status: closed" "$task_file" 2>/dev/null; then
        CLOSED_TASKS=$((CLOSED_TASKS + 1))
    elif grep -q "^status: active" "$task_file" 2>/dev/null; then
        IN_PROGRESS_TASKS=$((IN_PROGRESS_TASKS + 1))
        OPEN_TASKS=$((OPEN_TASKS + 1))
    else
        OPEN_TASKS=$((OPEN_TASKS + 1))
    fi
done

# Calculate progress
if [ "$TOTAL_TASKS" -gt 0 ]; then
    PROGRESS=$(( (CLOSED_TASKS * 100) / TOTAL_TASKS ))
else
    PROGRESS=0
fi

# Determine epic status
if [ "$PROGRESS" -eq 0 ]; then
    NEW_STATUS="backlog"
elif [ "$PROGRESS" -eq 100 ]; then
    NEW_STATUS="completed"
else
    NEW_STATUS="in-progress"
fi

# Find main epic file
EPIC_FILE=""
if [ -f "$EPIC_DIR/epic.md" ]; then
    EPIC_FILE="$EPIC_DIR/epic.md"
else
    # Look for numbered main task file
    for file in "$EPIC_DIR"/*.md; do
        if [[ -f "$file" ]] && [[ "$(basename "$file")" =~ ^[0-9]+\.md$ ]]; then
            EPIC_FILE="$file"
            break
        fi
    done
fi

if [ -z "$EPIC_FILE" ]; then
    echo "❌ No epic file found in $EPIC_DIR"
    exit 1
fi

# Get current values from epic file
OLD_PROGRESS=""
OLD_STATUS=""

if grep -q "^progress:" "$EPIC_FILE" 2>/dev/null; then
    OLD_PROGRESS=$(grep "^progress:" "$EPIC_FILE" | cut -d: -f2 | tr -d ' %')
fi

if grep -q "^status:" "$EPIC_FILE" 2>/dev/null; then
    OLD_STATUS=$(grep "^status:" "$EPIC_FILE" | cut -d: -f2 | tr -d ' ')
fi

# Update epic file frontmatter
CURRENT_DATETIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Create temp file for updates
TEMP_FILE=$(mktemp)
UPDATED=false

# Process the file line by line
while IFS= read -r line; do
    if [[ "$line" =~ ^status: ]]; then
        echo "status: $NEW_STATUS" >> "$TEMP_FILE"
        UPDATED=true
    elif [[ "$line" =~ ^progress: ]]; then
        echo "progress: $PROGRESS%" >> "$TEMP_FILE"
        UPDATED=true
    elif [[ "$line" =~ ^updated: ]]; then
        echo "updated: $CURRENT_DATETIME" >> "$TEMP_FILE"
        UPDATED=true
    else
        echo "$line" >> "$TEMP_FILE"
    fi
done < "$EPIC_FILE"

# If we didn't find existing fields in frontmatter, add them
if ! $UPDATED; then
    # Need to add fields to frontmatter - recreate file
    TEMP_FILE2=$(mktemp)
    IN_FRONTMATTER=false
    FRONTMATTER_ENDED=false
    
    while IFS= read -r line; do
        if [[ "$line" == "---" ]]; then
            echo "$line" >> "$TEMP_FILE2"
            if $IN_FRONTMATTER; then
                # End of frontmatter - add our fields before closing
                [ -n "$OLD_STATUS" ] || echo "status: $NEW_STATUS" >> "$TEMP_FILE2"
                echo "progress: $PROGRESS%" >> "$TEMP_FILE2"
                echo "updated: $CURRENT_DATETIME" >> "$TEMP_FILE2"
                FRONTMATTER_ENDED=true
            fi
            IN_FRONTMATTER=!$IN_FRONTMATTER
        else
            echo "$line" >> "$TEMP_FILE2"
        fi
    done < "$EPIC_FILE"
    
    mv "$TEMP_FILE2" "$TEMP_FILE"
fi

# Apply updates
mv "$TEMP_FILE" "$EPIC_FILE"

# Get GitHub issue number for task list update
EPIC_ISSUE=""
if grep -q "^github:" "$EPIC_FILE" 2>/dev/null; then
    EPIC_ISSUE=$(grep "^github:" "$EPIC_FILE" | grep -oE '[0-9]+$' || true)
fi

# Update GitHub task list if epic has GitHub issue
if [ ! -z "$EPIC_ISSUE" ] && command -v gh >/dev/null 2>&1; then
    echo "📋 Updating GitHub issue #$EPIC_ISSUE task list..."
    
    # Get current epic body
    if gh issue view "$EPIC_ISSUE" --json body -q .body > /tmp/epic-body.md 2>/dev/null; then
        
        # Update task checkboxes based on status
        for task_file in "$EPIC_DIR"/*.md; do
            if [[ ! -f "$task_file" ]] || [[ "$(basename "$task_file")" == "epic.md" ]]; then
                continue
            fi
            
            # Skip if not numbered task file
            if [[ ! "$(basename "$task_file")" =~ ^[0-9] ]]; then
                continue
            fi
            
            # Extract task GitHub issue number
            task_issue=$(grep '^github:' "$task_file" 2>/dev/null | grep -oE '[0-9]+$' || true)
            
            if [ ! -z "$task_issue" ]; then
                # Check task status
                if grep -q "^status: closed" "$task_file" 2>/dev/null; then
                    # Mark as checked
                    sed -i.bak "s/- \[ \] #$task_issue/- [x] #$task_issue/" /tmp/epic-body.md 2>/dev/null || true
                else
                    # Ensure unchecked
                    sed -i.bak "s/- \[x\] #$task_issue/- [ ] #$task_issue/" /tmp/epic-body.md 2>/dev/null || true
                fi
            fi
        done
        
        # Update epic issue
        if gh issue edit "$EPIC_ISSUE" --body-file /tmp/epic-body.md >/dev/null 2>&1; then
            echo "✅ GitHub task list updated"
        else
            echo "⚠️  Failed to update GitHub task list"
        fi
        
        # Cleanup
        rm -f /tmp/epic-body.md /tmp/epic-body.md.bak
    else
        echo "⚠️  Could not fetch GitHub issue #$EPIC_ISSUE"
    fi
fi

# Output results
echo
echo "📊 Epic refreshed: $EPIC_NAME"
echo
echo "Tasks:"
echo "  Closed: $CLOSED_TASKS"
echo "  Open: $OPEN_TASKS"
echo "  Total: $TOTAL_TASKS"
echo

if [ -n "$OLD_PROGRESS" ]; then
    echo "Progress: $OLD_PROGRESS% → $PROGRESS%"
else
    echo "Progress: $PROGRESS%"
fi

if [ -n "$OLD_STATUS" ]; then
    echo "Status: $OLD_STATUS → $NEW_STATUS"
else
    echo "Status: $NEW_STATUS"
fi

if [ ! -z "$EPIC_ISSUE" ]; then
    echo "GitHub: Task list updated ✓"
fi

echo

if [ "$PROGRESS" -eq 100 ]; then
    echo "🎉 Epic complete! Run /pm:epic-close $EPIC_NAME to close epic"
elif [ "$PROGRESS" -gt 0 ]; then
    echo "🔄 Epic in progress. Run /pm:next to see priority tasks"
else
    echo "📋 Epic ready to start. Run /pm:next to see available tasks"
fi