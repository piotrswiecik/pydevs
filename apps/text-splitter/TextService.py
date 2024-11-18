import re
class TextSplitter:
    def split(self, text: str, limit: int) -> list:
        print(f"Starting split process with limit: {limit} tokens")
        chunks = []
        position = 0
        total_length = len(text)
        current_headers = {}

        while position < total_length:
            print(f"Processing chunk starting at position: {position}")
            chunk_text, chunk_end = self.get_chunk(text, position, limit)
            tokens = self.count_tokens(chunk_text)
            print(f"Chunk tokens: {tokens}")

            headers_in_chunk = self.extract_headers(chunk_text)
            self.update_current_headers(current_headers, headers_in_chunk)

            content, urls, images = self.extract_urls_and_images(chunk_text)

            chunks.append({
                'text': content,
                'metadata': {
                    'tokens': tokens,
                    'headers': current_headers.copy(),
                    'urls': urls,
                    'images': images,
                },
            })

            position = chunk_end

        print(f"Split process completed. Total chunks: {len(chunks)}")
        return chunks
    
    def get_chunk(self, text: str, start: int, limit: int) -> tuple:
        print(f"Getting chunk starting at {start} with limit {limit}")

        # Account for token overhead due to formatting
        overhead = self.count_tokens(self.format_for_tokenization('')) - self.count_tokens('')

        # Initial tentative end position
        end = min(start + (len(text) - start) * limit // self.count_tokens(text[start:]), len(text))

        # Adjust end to avoid exceeding token limit
        chunk_text = text[start:end]
        tokens = self.count_tokens(chunk_text)

        while tokens + overhead > limit and end > start:
            print(f"Chunk exceeds limit with {tokens + overhead} tokens. Adjusting end position...")
            end = self.find_new_chunk_end(text, start, end)
            chunk_text = text[start:end]
            tokens = self.count_tokens(chunk_text)

        return chunk_text, end
    def extract_urls_and_images(self, text: str) -> dict:
        urls = []
        images = []
        url_index = 0
        image_index = 0

        # Replace image markdown and collect URLs
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', lambda match: self._replace_image(match, images, image_index), text)
        
        # Replace link markdown and collect URLs
        content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda match: self._replace_link(match, urls, url_index), content)

        return {'content': content, 'urls': urls, 'images': images}

    def _replace_image(self, match, images, image_index):
        alt_text = match.group(1)
        url = match.group(2)
        images.append(url)
        image_index += 1
        return f'![{alt_text}]{{{{img{image_index - 1}}}}}'

    def _replace_link(self, match, urls, url_index):
        link_text = match.group(1)
        url = match.group(2)
        urls.append(url)
        url_index += 1
        return f'[{link_text}]{{{{url{url_index - 1}}}}}'

    # Placeholder methods for the missing implementations
    # ...