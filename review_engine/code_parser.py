class CodeParser:
    def parse_files(self, files):
        changes = []
        for f in files:
            changes.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "patch": f.patch or ""  # raw diff
            })
        return changes
