#!/bin/bash

# Epic Sync - Push epic and tasks to GitHub as issues
# Usage: ./epic-sync.sh <epic_name>

set -e

EPIC_NAME="$1"

if [ -z "$EPIC_NAME" ]; then
    echo "Usage: $0 <epic_name>"
    exit 1
fi

EPIC_DIR=".claude/epics/$EPIC_NAME"
EPIC_FILE="$EPIC_DIR/epic.md"

echo "🔍 Checking epic: $EPIC_NAME"

# Verify epic exists
if [ ! -f "$EPIC_FILE" ]; then
    echo "❌ Epic not found. Run: /pm:prd-parse $EPIC_NAME"
    exit 1
fi

# Count task files
task_count=$(ls "$EPIC_DIR"/[0-9][0-9][0-9].md 2>/dev/null | wc -l | tr -d ' ')
if [ "$task_count" -eq 0 ]; then
    echo "❌ No tasks to sync. Run: /pm:epic-decompose $EPIC_NAME"
    exit 1
fi

echo "📊 Found $task_count tasks to sync"

# Check remote repository
remote_url=$(git remote get-url origin 2>/dev/null || echo "")
if [[ "$remote_url" == *"automazeio/ccpm"* ]]; then
    echo "❌ ERROR: You're trying to sync with the CCPM template repository!"
    echo ""
    echo "This repository (automazeio/ccpm) is a template for others to use."
    echo "You should NOT create issues or PRs here."
    echo ""
    echo "Current remote: $remote_url"
    exit 1
fi

echo "✅ Repository check passed"

# Create temp directory for processing
mkdir -p /tmp/epic-sync-$$

# 1. Create Epic Issue
echo "🚀 Creating epic issue..."

# Extract content without frontmatter
sed '1,/^---$/d; 1,/^---$/d' "$EPIC_FILE" > /tmp/epic-sync-$$/epic-body-raw.md

# Process Tasks Created section to Stats
awk '
  /^## Tasks Created/ {
    in_tasks=1
    next
  }
  /^## / && in_tasks && !/^## Tasks Created/ {
    in_tasks=0
    # Add Stats section
    if (total_tasks) {
      print "## Stats"
      print ""
      print "Total tasks: " total_tasks
      print "Parallel tasks: " parallel_tasks " (can be worked on simultaneously)"
      print "Sequential tasks: " sequential_tasks " (have dependencies)"
      if (total_effort) print "Estimated total effort: " total_effort
      print ""
    }
  }
  /^Total tasks:/ && in_tasks { total_tasks = $3; next }
  /^Parallel tasks:/ && in_tasks { parallel_tasks = $3; next }
  /^Sequential tasks:/ && in_tasks { sequential_tasks = $3; next }
  /^Estimated total effort:/ && in_tasks {
    gsub(/^Estimated total effort: /, "")
    total_effort = $0
    next
  }
  !in_tasks { print }
  END {
    # If we were still in tasks section at EOF, add stats
    if (in_tasks && total_tasks) {
      print "## Stats"
      print ""
      print "Total tasks: " total_tasks
      print "Parallel tasks: " parallel_tasks " (can be worked on simultaneously)"
      print "Sequential tasks: " sequential_tasks " (have dependencies)"
      if (total_effort) print "Estimated total effort: " total_effort
    }
  }
' /tmp/epic-sync-$$/epic-body-raw.md > /tmp/epic-sync-$$/epic-body.md

# Determine epic type
if grep -qi "bug\|fix\|issue\|problem\|error" /tmp/epic-sync-$$/epic-body.md; then
    epic_type="bug"
else
    epic_type="feature"
fi

# Create epic issue
epic_output=$(gh issue create \
    --title "Epic: $EPIC_NAME" \
    --body-file /tmp/epic-sync-$$/epic-body.md \
    --label "epic,$epic_type")
epic_number=$(echo "$epic_output" | grep -oE '[0-9]+$')

echo "✅ Created epic issue #$epic_number"

# Check if gh-sub-issue is available
if gh extension list | grep -q "yahsan2/gh-sub-issue"; then
    use_subissues=true
    echo "🔗 Using gh-sub-issue for task relationships"
else
    use_subissues=false
    echo "⚠️ gh-sub-issue not installed. Using fallback mode."
fi

# 2. Create Task Sub-Issues
echo "📝 Creating task issues..."

> /tmp/epic-sync-$$/task-mapping.txt

# Create tasks sequentially for now (can be enhanced for parallel later)
for task_file in "$EPIC_DIR"/[0-9][0-9][0-9].md; do
    [ -f "$task_file" ] || continue

    # Extract task name from frontmatter
    task_name=$(grep '^name:' "$task_file" | sed 's/^name: *//')

    # Strip frontmatter from task content
    sed '1,/^---$/d; 1,/^---$/d' "$task_file" > /tmp/epic-sync-$$/task-body.md

    echo "  Creating: $task_name"

    # Create sub-issue
    if [ "$use_subissues" = true ]; then
        task_body=$(cat /tmp/epic-sync-$$/task-body.md)
        task_output=$(gh sub-issue create \
            --parent "$epic_number" \
            --title "$task_name" \
            --body "$task_body" \
            --label "task")
        task_number=$(echo "$task_output" | grep -oE '#[0-9]+' | grep -oE '[0-9]+' | tail -1)
    else
        task_output=$(gh issue create \
            --title "$task_name" \
            --body-file /tmp/epic-sync-$$/task-body.md \
            --label "task")
        task_number=$(echo "$task_output" | grep -oE '[0-9]+$')
    fi

    # Record mapping
    echo "$task_file:$task_number" >> /tmp/epic-sync-$$/task-mapping.txt
    echo "  ✅ Created task issue #$task_number"
done

# 3. Rename Task Files and Update References
echo "🔄 Updating task references..."

# Build mapping from old numbers to new issue IDs
> /tmp/epic-sync-$$/id-mapping.txt
while IFS=: read -r task_file task_number; do
    old_num=$(basename "$task_file" .md)
    echo "$old_num:$task_number" >> /tmp/epic-sync-$$/id-mapping.txt
done < /tmp/epic-sync-$$/task-mapping.txt

# Get repo info for GitHub URLs
repo=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || git remote get-url origin | sed 's|.*github.com[:/]||' | sed 's|\.git$||')
current_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Process each task file
while IFS=: read -r task_file task_number; do
    new_name="$(dirname "$task_file")/${task_number}.md"

    # Read the file content
    content=$(cat "$task_file")

    # Update depends_on and conflicts_with references
    while IFS=: read -r old_num new_num; do
        content=$(echo "$content" | sed "s/\b$old_num\b/$new_num/g")
    done < /tmp/epic-sync-$$/id-mapping.txt

    # Update github field in frontmatter
    github_url="https://github.com/$repo/issues/$task_number"
    
    # Update frontmatter
    content=$(echo "$content" | sed "s|^github:.*|github: $github_url|")
    content=$(echo "$content" | sed "s|^updated:.*|updated: $current_date|")

    # Write updated content to new file
    echo "$content" > "$new_name"

    # Remove old file if different from new
    [ "$task_file" != "$new_name" ] && rm "$task_file"

    echo "  Renamed: $(basename "$task_file") → $(basename "$new_name")"
done < /tmp/epic-sync-$$/task-mapping.txt

# 4. Update Epic with Task List (fallback only)
if [ "$use_subissues" = false ]; then
    echo "📋 Adding task list to epic issue..."
    
    # Get current epic body 
    gh issue view "$epic_number" --json body -q .body > /tmp/epic-sync-$$/epic-current.md 2>/dev/null || gh issue view "$epic_number" | sed -n '/--$/,$p' | tail -n +2 > /tmp/epic-sync-$$/epic-current.md
    
    # Append task list
    echo "" >> /tmp/epic-sync-$$/epic-current.md
    echo "## Tasks" >> /tmp/epic-sync-$$/epic-current.md
    
    for task_file in "$EPIC_DIR"/[0-9]*.md; do
        [ -f "$task_file" ] || continue
        issue_num=$(basename "$task_file" .md)
        task_name=$(grep '^name:' "$task_file" | sed 's/^name: *//')
        echo "- [ ] #${issue_num} ${task_name}" >> /tmp/epic-sync-$$/epic-current.md
    done
    
    # Update epic issue
    gh issue edit "$epic_number" --body-file /tmp/epic-sync-$$/epic-current.md
fi

# 5. Update Epic File
echo "📄 Updating epic file..."

epic_url="https://github.com/$repo/issues/$epic_number"

# Update epic frontmatter
sed -i.bak "s|^github:.*|github: $epic_url|" "$EPIC_FILE"
sed -i.bak "s|^updated:.*|updated: $current_date|" "$EPIC_FILE"
rm "${EPIC_FILE}.bak"

# Update Tasks Created section with real issue numbers
cat > /tmp/epic-sync-$$/tasks-section.md << 'EOF'
## Tasks Created
EOF

for task_file in "$EPIC_DIR"/[0-9]*.md; do
    [ -f "$task_file" ] || continue

    issue_num=$(basename "$task_file" .md)
    task_name=$(grep '^name:' "$task_file" | sed 's/^name: *//')
    parallel=$(grep '^parallel:' "$task_file" | sed 's/^parallel: *//')

    echo "- [ ] #${issue_num} - ${task_name} (parallel: ${parallel})" >> /tmp/epic-sync-$$/tasks-section.md
done

# Add summary statistics
total_count=$(ls "$EPIC_DIR"/[0-9]*.md 2>/dev/null | wc -l | tr -d ' ')
parallel_count=$(grep -l '^parallel: true' "$EPIC_DIR"/[0-9]*.md 2>/dev/null | wc -l | tr -d ' ')
sequential_count=$((total_count - parallel_count))

cat >> /tmp/epic-sync-$$/tasks-section.md << EOF

Total tasks: ${total_count}
Parallel tasks: ${parallel_count}
Sequential tasks: ${sequential_count}
EOF

# Replace the Tasks Created section in epic.md
cp "$EPIC_FILE" "${EPIC_FILE}.backup"

awk '
  /^## Tasks Created/ {
    skip=1
    while ((getline line < "/tmp/epic-sync-'"$$"'/tasks-section.md") > 0) print line
    close("/tmp/epic-sync-'"$$"'/tasks-section.md")
  }
  /^## / && !/^## Tasks Created/ { skip=0 }
  !skip && !/^## Tasks Created/ { print }
' "${EPIC_FILE}.backup" > "$EPIC_FILE"

rm "${EPIC_FILE}.backup"

# 6. Create Mapping File
cat > "$EPIC_DIR/github-mapping.md" << EOF
# GitHub Issue Mapping

Epic: #${epic_number} - https://github.com/${repo}/issues/${epic_number}

Tasks:
EOF

for task_file in "$EPIC_DIR"/[0-9]*.md; do
    [ -f "$task_file" ] || continue

    issue_num=$(basename "$task_file" .md)
    task_name=$(grep '^name:' "$task_file" | sed 's/^name: *//')

    echo "- #${issue_num}: ${task_name} - https://github.com/${repo}/issues/${issue_num}" >> "$EPIC_DIR/github-mapping.md"
done

echo "" >> "$EPIC_DIR/github-mapping.md"
echo "Synced: $current_date" >> "$EPIC_DIR/github-mapping.md"

# 7. Create Worktree (optional - commented out for now)
# echo "🌳 Creating development worktree..."
# git checkout main >/dev/null 2>&1 || true
# git pull origin main >/dev/null 2>&1 || true
# git worktree add "../epic-$EPIC_NAME" -b "epic/$EPIC_NAME" >/dev/null 2>&1 || true

# Clean up
rm -rf /tmp/epic-sync-$$

echo
echo "✅ Synced to GitHub"
echo "  - Epic: #$epic_number - Epic: $EPIC_NAME"  
echo "  - Tasks: $task_count sub-issues created"
echo "  - Labels applied: epic, task, epic:$EPIC_NAME"
echo "  - Files renamed: 001.md → {issue_id}.md"
echo "  - References updated: depends_on/conflicts_with now use issue IDs"
echo
echo "Next steps:"
echo "  - View epic: https://github.com/$repo/issues/$epic_number"
echo "  - Work on tasks: /pm:next"
echo