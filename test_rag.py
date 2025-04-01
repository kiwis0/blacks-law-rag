import unittest
import os
from rag import index_document, query_rag, chunks, metadata
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class TestRAGSystem(unittest.TestCase):
    def setUp(self):
        self.test_pdf = "test_content.pdf"
        c = canvas.Canvas(self.test_pdf, pagesize=letter)
        # Simulate two-column layout with distinct positions
        textobject = c.beginText()
        textobject.setTextOrigin(50, 750)  # Left column
        textobject.textLines("AFFINAGE. A refining of metals. Blount.")
        textobject.setTextOrigin(300, 750)  # Right column
        textobject.textLines("AFFIDAVIT. A written or printed declaration or statement of facts, made voluntarily, and confirmed by the oath or affirmation.")
        c.drawText(textobject)
        c.showPage()
        c.save()

    def test_index_document(self):
        index_document(self.test_pdf)
        self.assertTrue(os.path.exists("blacks_law_index.bin"))
        self.assertTrue(os.path.exists("blacks_law_chunks.pkl"))
        self.assertTrue(os.path.exists("blacks_law_metadata.pkl"))
        self.assertTrue(len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}: {chunks}")
        self.assertTrue(len(metadata) == len(chunks))
        self.assertIn("AFFINAGE", metadata)
        self.assertIn("AFFIDAVIT", metadata)

    def test_query_rag(self):
        index_document(self.test_pdf)
        answer = query_rag("What does 'affidavit' mean?")
        self.assertIsInstance(answer, str)
        self.assertIn("written", answer.lower())
        answer = query_rag("What does 'affinage' mean?")
        self.assertIn("refining", answer.lower())

    def tearDown(self):
        for file in ["test_content.pdf", "blacks_law_index.bin", "blacks_law_chunks.pkl", "blacks_law_metadata.pkl"]:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    unittest.main()