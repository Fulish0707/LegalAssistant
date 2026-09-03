import unittest


class HtmlSanitizerTests(unittest.TestCase):
    def test_removes_executable_markup_but_preserves_report_structure(self):
        from html_sanitizer import sanitize_report_html

        result = sanitize_report_html(
            '<h2>Summary</h2><ul><li>Point</li></ul>'
            '<table><tr><td>42</td></tr></table>'
            '<a href="https://example.com">Source</a>'
            '<script>alert(1)</script><img src=x onerror="alert(2)">'
            '<a href="javascript:alert(3)">Bad</a><iframe src="https://bad.example"></iframe>'
            '<p style="color:red">Styled</p>'
        )

        self.assertIn('<h2>Summary</h2>', result)
        self.assertIn('<ul><li>Point</li></ul>', result)
        self.assertIn('<table><tbody><tr><td>42</td></tr></tbody></table>', result)
        self.assertIn('href="https://example.com"', result)
        self.assertNotIn('<script', result)
        self.assertNotIn('onerror', result)
        self.assertNotIn('javascript:', result)
        self.assertNotIn('<iframe', result)
        self.assertNotIn('style=', result)

