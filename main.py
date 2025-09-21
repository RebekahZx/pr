<<<<<<< HEAD
import argparse

from review_engine.git_handler import GitHandler

from review_engine.code_parser import CodeParser

from review_engine.reviewer import Reviewer



def main():

    parser = argparse.ArgumentParser(description="PR Review Agent MVP")

    parser.add_argument("--token", required=True, help="GitHub personal access token")

    parser.add_argument("--repo", required=True, help="Repo name e.g. user/repo")

    parser.add_argument("--pr", type=int, required=True, help="Pull request number")

    args = parser.parse_args()



    # Step 1: Fetch PR files

    gh = GitHandler(args.token, args.repo)

    files = gh.get_pr_diff(args.pr)



    # Step 2: Parse changes

    code_parser = CodeParser()

    changes = code_parser.parse_files(files)



    # Step 3: Rule-based review

    reviewer = Reviewer()

    feedback = reviewer.review(changes)



    # Step 4: Print results

    print("\n=== PR Review Report ===")

    if not feedback:

        print("✅ No issues found!")

    else:

        for fb in feedback:

            print(f" - {fb}")
    # after you compute 'changes' and basic_feedback

from review_engine.ai_reviewer import AIReviewer

# instantiate
ai = AIReviewer(model="gemini-2.5-flash", max_tokens=512)

print("\n=== Basic Rule-based Review ===")
# ... existing printing ...

print("\n=== AI Review (Gemini) ===")
for c in changes:
    filename = c["filename"]
    patch = c["patch"] or ""
    # skip tiny/empty patches
    if not patch.strip():
        print(f" - {filename}: (no patch to AI review)")
        continue

    # optionally pass repo conventions string (short) to help reviewer
    repo_ctx = "Language: python. Use logging not print. Follow PEP8." if filename.endswith(".py") else None

    ai_resp = ai.review_patch(filename, patch, role_ctx=repo_ctx)
    print(f"\n--- {filename} ---\n{ai_resp}\n")




if __name__ == "__main__":

    main()

=======
import argparse

from review_engine.git_handler import GitHandler

from review_engine.code_parser import CodeParser

from review_engine.reviewer import Reviewer



def main():

    parser = argparse.ArgumentParser(description="PR Review Agent MVP")

    parser.add_argument("--token", required=True, help="GitHub personal access token")

    parser.add_argument("--repo", required=True, help="Repo name e.g. user/repo")

    parser.add_argument("--pr", type=int, required=True, help="Pull request number")

    args = parser.parse_args()



    # Step 1: Fetch PR files

    gh = GitHandler(args.token, args.repo)

    files = gh.get_pr_diff(args.pr)



    # Step 2: Parse changes

    code_parser = CodeParser()

    changes = code_parser.parse_files(files)



    # Step 3: Rule-based review

    reviewer = Reviewer()

    feedback = reviewer.review(changes)



    # Step 4: Print results

    print("\n=== PR Review Report ===")

    if not feedback:

        print("✅ No issues found!")

    else:

        for fb in feedback:

            print(f" - {fb}")
    # after you compute 'changes' and basic_feedback

from review_engine.ai_reviewer import AIReviewer

# instantiate
ai = AIReviewer(model="gemini-2.5-flash", max_tokens=512)

print("\n=== Basic Rule-based Review ===")
# ... existing printing ...

print("\n=== AI Review (Gemini) ===")
for c in changes:
    filename = c["filename"]
    patch = c["patch"] or ""
    # skip tiny/empty patches
    if not patch.strip():
        print(f" - {filename}: (no patch to AI review)")
        continue

    # optionally pass repo conventions string (short) to help reviewer
    repo_ctx = "Language: python. Use logging not print. Follow PEP8." if filename.endswith(".py") else None

    ai_resp = ai.review_patch(filename, patch, role_ctx=repo_ctx)
    print(f"\n--- {filename} ---\n{ai_resp}\n")




if __name__ == "__main__":

    main()

>>>>>>> bd08937 (Initial commit)
