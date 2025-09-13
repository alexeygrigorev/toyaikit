from toyaikit.utils import strip_matching_outer_html_tags


class TestStripMatchingOuterHtmlTags:
    def test_strip_basic_tags(self):
        """Test stripping basic HTML tags."""
        input_text = "<div>Hello World</div>"
        expected = "Hello World"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_strip_tags_with_attributes(self):
        """Test stripping tags with attributes."""
        input_text = '<p class="test" id="paragraph">Content here</p>'
        expected = "Content here"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_strip_tags_with_whitespace(self):
        """Test stripping tags with surrounding whitespace."""
        input_text = "  <span>  Some text  </span>  "
        expected = "Some text"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_multiline_content(self):
        """Test stripping tags with multiline content."""
        input_text = """<div>
            Line 1
            Line 2
            Line 3
        </div>"""
        expected = "Line 1\n            Line 2\n            Line 3"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_nested_tags_only_strips_outer(self):
        """Test that only outer tags are stripped, inner tags remain."""
        input_text = "<div><span>Inner content</span></div>"
        expected = "<span>Inner content</span>"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_mismatched_tags_returns_original(self):
        """Test that mismatched tags return original text stripped."""
        input_text = "<div>Content</span>"
        expected = "<div>Content</span>"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_no_tags_returns_stripped_text(self):
        """Test that text without tags is just stripped of whitespace."""
        input_text = "  Just plain text  "
        expected = "Just plain text"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_empty_string(self):
        """Test empty string handling."""
        input_text = ""
        expected = ""
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_whitespace_only(self):
        """Test whitespace-only string."""
        input_text = "   \n  \t  "
        expected = ""
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_self_closing_tags_no_match(self):
        """Test that self-closing tags don't match."""
        input_text = "<img src='test.jpg' />"
        expected = "<img src='test.jpg' />"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_incomplete_opening_tag(self):
        """Test incomplete opening tag."""
        input_text = "<div"
        expected = "<div"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_incomplete_closing_tag(self):
        """Test incomplete closing tag."""
        input_text = "<div>Content</div"
        expected = "<div>Content</div"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_multiple_outer_tags_no_match(self):
        """Test multiple outer tags don't match."""
        input_text = "<div>Content</div><span>More</span>"
        expected = "<div>Content</div><span>More</span>"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_complex_attributes(self):
        """Test tags with complex attributes."""
        input_text = '<div class="container" data-id="123" style="color: red;">Complex content</div>'
        expected = "Complex content"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_tags_with_numbers(self):
        """Test tags with numbers in name."""
        input_text = "<h1>Heading</h1>"
        expected = "Heading"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_case_sensitive_tags(self):
        """Test that tag matching is case sensitive."""
        input_text = "<DIV>Content</div>"
        expected = "<DIV>Content</div>"  # Should not match due to case difference
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_empty_tags(self):
        """Test empty tags."""
        input_text = "<div></div>"
        expected = ""
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_tags_with_special_characters_in_content(self):
        """Test content with special characters."""
        input_text = "<p>Special chars: &amp; &lt; &gt; &quot;</p>"
        expected = "Special chars: &amp; &lt; &gt; &quot;"
        assert strip_matching_outer_html_tags(input_text) == expected

    def test_deeply_nested_content(self):
        """Test deeply nested HTML structure."""
        input_text = (
            "<article><section><div><p>Deep content</p></div></section></article>"
        )
        expected = "<section><div><p>Deep content</p></div></section>"
        assert strip_matching_outer_html_tags(input_text) == expected
