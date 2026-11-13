export type User={email:string;name:string;role:string};
export type Dataset={id:string;name:string;source:string;row_count:number;characters:number};
export type Lexicon={id:string;name:string;version:string;terms:string[];stopwords:string[]};
export type Study={id:string;name:string;dataset_id:string;lexicon_id:string;topic_count:number;max_iter:number;tolerance:number;seed:number};
export type Topic={id:string;top_terms:{term:string;weight:number;display_fraction:number}[]};
export type Document={doc_id:string;excerpt:string;weights:number[];display_fractions:number[];unknown_characters:number};
export type Result={documents:Document[];document_count:number;included:number;isolated:{doc_id:string;reason:string}[];feature_count:number;topics:Topic[];neighbors:{doc_id:string;neighbors:{doc_id:string;similarity:number}[]}[];nmf:{w:number[][];h:number[][];topic_count:number;iterations:number;converged:boolean;relative_change:number;error:number;relative_error:number;error_history:{iteration:number;error:number}[];seed:number;warning:string};lexicon_snapshot:Lexicon;study:Study;checksum:string;limitations:string[]};
export type Run={id:string;study_id:string;status:string;result:Result;error:string};
export type Review={id:string;run_id:string;comment:string;theme_names:Record<string,string>;status:string;reviewer:string};
export type Report={id:string;run_id:string;review_id:string;title:string};

export type ProjectionDocument={doc_id:string;token_count:number;unknown_characters:number;out_of_vocabulary_terms:string[];weights:number[];display_fractions:number[];iterations:number;converged:boolean;relative_change:number;reconstruction_error:number;relative_reconstruction_error:number;neighbors:{doc_id:string;similarity:number}[]};
export type ProjectionIsolated={doc_id:string;reason:string;token_count:number;unknown_characters:number;out_of_vocabulary_terms:string[]};
export type Projection={run_id:string;status:string;basis:{lexicon:{id:string;name:string;version:string;term_count:number;stopword_count:number};topic_count:number;topic_ids:string[];vocabulary_size:number;vocabulary_checksum:string;idf_source:string;vocabulary_order:string;original_document_count:number;basis_checksum:string};documents:ProjectionDocument[];isolated:ProjectionIsolated[];summary:{input_documents:number;included:number;isolated:number;aggregate_reconstruction_error:number;aggregate_relative_reconstruction_error:number;max_relative_reconstruction_error:number;topic_count:number;neighbor_limit:number};projection:{method:string;max_iterations:number;minimum_iterations:number;tolerance:number;initialization:string;neighbor_limit:number};limitations:string[]};
