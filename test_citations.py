
import rag
import config

hits = [
    {
        "text": "CSE tuition fee is Rs. 160000.",
        "source": "02_Fee_Structure.pdf",
        "page": 1,
        "score": 0.85
    },
    {
        "text": "Double room costs Rs. 80000 per year.",
        "source": "03_Hostel_Rules.pdf",
        "page": 1,
        "score": 0.78
    }
]

# Test 1: Valid citations
valid_response = """
{
    "answer": "CSE tuition is Rs. 160000.",
    "citations": ["S1"]
}
"""

result = rag._parse_grounded_response(valid_response, hits)

assert result is not None
assert result["sources"][0]["source"] == "02_Fee_Structure.pdf"

print("PASS: Valid citation accepted")

# Test 2: Fabricated citation
invalid_response = """
{
    "answer": "CSE tuition is Rs. 160000.",
    "citations": ["S9"]
}
"""

result = rag._parse_grounded_response(invalid_response, hits)

assert result is None

print("PASS: Fabricated citation rejected")

# Test 3: Missing citations
missing_response = """
{
    "answer": "CSE tuition is Rs. 160000.",
    "citations": []
}
"""

result = rag._parse_grounded_response(missing_response, hits)

assert result is None

print("PASS: Missing citations rejected")

print("\nAll citation validation tests passed!")
