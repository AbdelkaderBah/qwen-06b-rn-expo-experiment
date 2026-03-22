type AnyComponent = import('react').ComponentType<any>;
type AnyModifier = (...args: any[]) => unknown;

declare module 'expo-server' {
  export function origin(): string | null | undefined;
  export function environment(): string | null | undefined;
  export function runTask(task: () => void | Promise<void>): void;
  export function deferTask(task: () => void | Promise<void>): void;
  export function setResponseHeaders(headers: HeadersInit | ((headers: Headers) => void)): void;

  export class StatusError extends Error {
    constructor(status: number, message?: string);
    status: number;
  }
}

declare module 'expo-server/adapter/*' {
  export function createRequestHandler(
    options: Record<string, unknown>
  ): (request: Request) => Response | Promise<Response>;
}

declare module '@expo/ui/swift-ui' {
  export const Host: AnyComponent;
  export const Text: AnyComponent;
  export const Button: AnyComponent;
  export const VStack: AnyComponent;
  export const Section: AnyComponent;
  export const List: AnyComponent;
  export const BottomSheet: AnyComponent;
  export const Image: AnyComponent;
  export const Toggle: AnyComponent;
  export const DatePicker: AnyComponent;
  export const HStack: AnyComponent;
  export const Menu: AnyComponent;
  export const Picker: AnyComponent;
  export const ContextMenu: AnyComponent;
  export const TextField: AnyComponent;
  export const Form: AnyComponent;
  export const Divider: AnyComponent;
  export const RNHostView: AnyComponent;
  export const ProgressView: AnyComponent;
  export const Label: AnyComponent;
  export const Gauge: AnyComponent;
  export const Group: AnyComponent;
}

declare module '@expo/ui/swift-ui/modifiers' {
  export const foregroundStyle: AnyModifier;
  export const padding: AnyModifier;
  export const font: AnyModifier;
  export const tag: AnyModifier;
  export const frame: AnyModifier;
  export const pickerStyle: AnyModifier;
  export const presentationDetents: AnyModifier;
  export const tint: AnyModifier;
  export const background: AnyModifier;
  export const labelStyle: AnyModifier;
  export const presentationDragIndicator: AnyModifier;
  export const buttonStyle: AnyModifier;
  export const cornerRadius: AnyModifier;
  export const refreshable: AnyModifier;
  export const datePickerStyle: AnyModifier;
  export const bold: AnyModifier;
  export const italic: AnyModifier;
  export const shadow: AnyModifier;
  export const foregroundColor: AnyModifier;
  export const onTapGesture: AnyModifier;
  export const gaugeStyle: AnyModifier;
}

declare module '@expo/ui/jetpack-compose' {
  export const Host: AnyComponent;
  export const Text: AnyComponent;
  export const Button: AnyComponent;
  export const Column: AnyComponent;
  export const Icon: AnyComponent;
  export const Row: AnyComponent;
  export const ListItem: AnyComponent;
  export const Box: AnyComponent;
  export const ModalBottomSheet: AnyComponent;
  export const Slider: AnyComponent;
  export const Shape: AnyComponent;
  export const LazyColumn: AnyComponent;
  export const DateTimePicker: AnyComponent;
  export const TextButton: AnyComponent;
  export const Switch: AnyComponent;
  export const TextInput: AnyComponent;
  export const Surface: AnyComponent;
  export const Spacer: AnyComponent;
  export const IconButton: AnyComponent;
  export const AlertDialog: AnyComponent;
  export const CircularProgressIndicator: AnyComponent;
  export const Checkbox: AnyComponent;
  export const SegmentedButton: AnyComponent;
  export const ToggleButton: AnyComponent;
  export const FlowRow: AnyComponent;
}

declare module '@expo/ui/jetpack-compose/modifiers' {
  export const paddingAll: AnyModifier;
  export const size: AnyModifier;
  export const background: AnyModifier;
  export const fillMaxWidth: AnyModifier;
  export const height: AnyModifier;
  export const padding: AnyModifier;
  export const clip: AnyModifier;
  export const Shapes: AnyModifier;
  export const width: AnyModifier;
  export const weight: AnyModifier;
  export const clickable: AnyModifier;
  export const border: AnyModifier;
  export const shadow: AnyModifier;
  export const fillMaxSize: AnyModifier;
  export const offset: AnyModifier;
  export const selectable: AnyModifier;
  export const align: AnyModifier;
  export const alpha: AnyModifier;
  export const blur: AnyModifier;
  export const rotate: AnyModifier;
  export const zIndex: AnyModifier;
  export const animateContentSize: AnyModifier;
  export const matchParentSize: AnyModifier;
  export const testID: AnyModifier;
  export const toggleable: AnyModifier;
}

declare module '@expo/ui/datetimepicker' {
  const DateTimePicker: AnyComponent;
  export default DateTimePicker;
}

declare module 'react-native-markdown-display' {
  const MarkdownDisplay: AnyComponent;
  export default MarkdownDisplay;
}

declare module 'react-native-markdown-editor' {
  const MarkdownEditor: AnyComponent;
  export default MarkdownEditor;
}
