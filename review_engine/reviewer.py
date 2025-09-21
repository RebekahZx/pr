class Reviewer:
    def review(self, changes):
        feedback = []
        for c in changes:
            filename = c["filename"]

            # Rule 1: Too many additions
            if c["additions"] > 200:
                feedback.append(f"⚠️ {filename}: Large file change with {c['additions']} additions. Consider splitting the PR.")

            # Rule 2: Print statements
            if "print(" in c["patch"]:
                feedback.append(f"🛑 {filename}: Found print statements. Use logging instead.")

            # Rule 3: TODO/FIXME
            if "TODO" in c["patch"] or "FIXME" in c["patch"]:
                feedback.append(f"⚠️ {filename}: Contains TODO/FIXME comments. Resolve before merging.")

            # Rule 4: Empty file
            if c["additions"] == 0 and c["deletions"] == 0:
                feedback.append(f"ℹ️ {filename}: No actual changes.")

        return feedback
