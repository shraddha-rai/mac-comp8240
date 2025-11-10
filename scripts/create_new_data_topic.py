# pip install wikipedia-api==0.5.4
import wikipediaapi
import random
import math

wiki = wikipediaapi.Wikipedia('en')

# Mapping topics to Wikipedia categories
topic_to_category = {
    "Society & Culture": "Category:Society",
    "Science & Mathematics": "Category:Science",
    "Health": "Category:Health",
    "Education & Reference": "Category:Education",
    "Computers & Internet": "Category:Computing",
    "Sports": "Category:Sports",
    "Business & Finance": "Category:Business",
    "Entertainment & Music": "Category:Entertainment",
    "Family & Relationships": "Category:Family",
    "Politics & Government": "Category:Politics"
}

TARGET_LINES_PER_CATEGORY = 2000  # target 20k lines total for 10 categories
SNIPPET_WORDS = 50                # words per snippet
SNIPPETS_PER_LINE = 6             # 6 snippets per line

def get_category_members(category_name, max_pages=1000):
    cat_page = wiki.page(category_name)
    pages = []
    count = 0
    for _, member in cat_page.categorymembers.items():
        if member.ns == 0:
            pages.append(member)
            count += 1
            if count >= max_pages:
                break
    return pages

def sample_snippets(text, words_per_snippet, total_snippets):
    words = text.split()
    snippets = []
    if len(words) <= words_per_snippet:
        return [" ".join(words)]
    max_start = len(words) - words_per_snippet
    for _ in range(total_snippets):
        start = random.randint(0, max_start)
        snippet = " ".join(words[start:start+words_per_snippet])
        snippets.append(snippet)
    return snippets

output_file = "wikipedia_topics_dataset.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for topic, category_name in topic_to_category.items():
        print(f"Processing category: {category_name} for topic '{topic}' ...")
        pages = get_category_members(category_name, max_pages=1000)
        num_pages = len(pages)
        if num_pages == 0:
            continue

        # Calculate snippets per page to meet target lines
        total_snippets_needed = TARGET_LINES_PER_CATEGORY * SNIPPETS_PER_LINE
        snippets_per_page = math.ceil(total_snippets_needed / num_pages)
        print(f"  {num_pages} pages, sampling {snippets_per_page} snippets per page")

        all_snippets = []
        for page in pages:
            page_snippets = sample_snippets(page.text, SNIPPET_WORDS, snippets_per_page)
            all_snippets.extend(page_snippets)

        # Shuffle all snippets to increase randomness
        random.shuffle(all_snippets)

        # Combine snippets into lines of 6 snippets each
        for i in range(0, len(all_snippets), SNIPPETS_PER_LINE):
            combined = " ".join(all_snippets[i:i+SNIPPETS_PER_LINE])
            if combined.strip():
                f.write(f"{topic}\t{combined}\n")

print(f"Dataset saved as {output_file}")