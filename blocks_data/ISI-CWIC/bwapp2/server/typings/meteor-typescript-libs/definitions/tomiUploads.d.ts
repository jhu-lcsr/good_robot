/// <reference path="node.d.ts" />

interface UploadServerStatic {
	init(config: {
		tmpDir?: string;
		uploadDir?: string;
		checkCreateDirectories?: boolean;
        getDirectory: (fileInfo: any, formData: any) => string;
        getFileName: (fileInfo: any, formData: any) => string;
        finished: (fileInfo: any, formData: any) => void;
	}) : void;
}

declare var UploadServer : UploadServerStatic;

declare var process: NodeJS.Process;  // redundant with Node def, but want to call out this dependency
