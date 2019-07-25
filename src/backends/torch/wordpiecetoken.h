/**
 * DeepDetect
 * Copyright (c) 2019 Jolibrain
 * Author: Louis Jean <ljean@etud.insa-toulouse.fr>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef WORDPIECETOKEN_H
#define WORDPIECETOKEN_H

#include <string>
#include <vector>

#include <boost/tokenizer.hpp>

namespace dd {

/** Tokenizer that uses greedy longest-match-first search to cut words
 * in pieces */
class WordPieceTokenizer {
public:
    std::vector<std::string> _tokens;


    WordPieceTokenizer() {}

    template <typename VocabType>
    void tokenize(const std::string &input, const VocabType& vocab)
    {
        _tokens.clear();
        boost::char_separator<char> sep("\n\t\f\r ");
        boost::tokenizer<boost::char_separator<char>> base_tokens(input, sep);

        for (std::string word : base_tokens)
        {
            tokenize_word(word, vocab);
        }
    }

private:
    template <typename VocabType>
    void tokenize_word(const std::string &word, const VocabType& vocab)
    {
        int start = 0;
        std::vector<std::string> subtokens;
        bool is_bad = false;

        while (start < word.size())
        {
            int end = word.size();
            std::string cur_substr = "";
            while (start < end)
            {
                std::string substr = word.substr(start, end - start);
                if (start > 0)
                {
                    substr = _prefix + substr;
                }
                if (in_vocab(substr, vocab))
                {
                    cur_substr = substr;
                    break;
                }
                --end;
            }
            if (cur_substr == "")
            {
                is_bad = true;
                break;
            }
            subtokens.push_back(cur_substr);
            start = end;
        }

        if (is_bad)
        {
            _tokens.push_back(_unk_token);
        }
        else
        {
            _tokens.insert(_tokens.end(), subtokens.begin(), subtokens.end());
        }
    }

    template <typename VocabType>
    bool in_vocab(const std::string &tok, const VocabType& vocab)
    {
        return vocab.find(tok) != vocab.end();
    }

    std::string _prefix = "##";
    std::string _unk_token = "[UNK]";
};

}

#endif // WORDPIECETOKEN_H